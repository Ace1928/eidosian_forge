import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
class MTurkDataHandler:
    """
    Handles logging data to and reading data from a SQLite3 table for observation across
    processes and for controlled restarts.
    """

    def __init__(self, task_group_id=None, file_name='pmt_data.db'):
        self.db_path = os.path.join(data_dir, file_name)
        self.conn = {}
        self.task_group_id = task_group_id
        self.table_access_condition = threading.Condition()
        self.create_default_tables()

    @staticmethod
    def save_world_data(prepped_save_data, task_group_id, conversation_id, sandbox=False):
        target = 'sandbox' if sandbox else 'live'
        if task_group_id is None:
            return
        target_dir = os.path.join(data_dir, target, task_group_id, conversation_id)
        custom_data = prepped_save_data['custom_data']
        if custom_data is not None:
            target_dir_custom = os.path.join(target_dir, 'custom')
            if custom_data.get('needs-pickle') is not None:
                pickle_file = os.path.join(target_dir_custom, 'data.pickle')
                force_dir(pickle_file)
                with open(pickle_file, 'wb') as outfile:
                    pickle.dump(custom_data['needs-pickle'], outfile)
                del custom_data['needs-pickle']
            custom_file = os.path.join(target_dir_custom, 'data.json')
            force_dir(custom_file)
            print('Saving data to {}.'.format(custom_file))
            with open(custom_file, 'w') as outfile:
                json.dump(custom_data, outfile)
        worker_data = prepped_save_data['worker_data']
        target_dir_workers = os.path.join(target_dir, 'workers')
        for worker_id, w_data in worker_data.items():
            worker_file = os.path.join(target_dir_workers, '{}.json'.format(worker_id))
            force_dir(worker_file)
            with open(worker_file, 'w') as outfile:
                json.dump(w_data, outfile)

    def _get_connection(self):
        """
        Returns a singular database connection to be shared amongst all calls.
        """
        curr_thread = threading.get_ident()
        if curr_thread not in self.conn or self.conn[curr_thread] is None:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                self.conn[curr_thread] = conn
            except sqlite3.Error as e:
                shared_utils.print_and_log(logging.ERROR, 'Could not get db connection, failing: {}'.format(repr(e)), should_print=True)
                raise e
        return self.conn[curr_thread]

    def _force_task_group_id(self, task_group_id):
        """
        Throw an error if a task group id is neither provided nor stored.
        """
        if task_group_id is None:
            task_group_id = self.task_group_id
        assert task_group_id is not None, 'Default task_group_id not set'
        return task_group_id

    def create_default_tables(self):
        """
        Prepares the default tables in the database if they don't exist.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute(CREATE_RUN_DATA_SQL_TABLE)
            c.execute(CREATE_WORKER_DATA_SQL_TABLE)
            c.execute(CREATE_HIT_DATA_SQL_TABLE)
            c.execute(CREATE_ASSIGN_DATA_SQL_TABLE)
            c.execute(CREATE_PAIRING_DATA_SQL_TABLE)
            updates = ['ALTER TABLE pairings\n                   ADD COLUMN onboarding_id TEXT default null', 'ALTER TABLE runs\n                   ADD COLUMN taskname TEXT default null', 'ALTER TABLE runs\n                   ADD COLUMN launch_time int default 0', 'ALTER TABLE pairings\n                   ADD COLUMN extra_bonus_amount INT default 0', "ALTER TABLE pairings\n                   ADD COLUMN extra_bonus_text TEXT default '';"]
            for update in updates:
                try:
                    c.execute(update)
                except sqlite3.Error:
                    pass
            conn.commit()

    def log_new_run(self, target_hits, taskname, task_group_id=None):
        """
        Add a new run to the runs table.
        """
        with self.table_access_condition:
            task_group_id = self._force_task_group_id(task_group_id)
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?);', (task_group_id, 0, target_hits, 0, 0, taskname, time.time()))
            conn.commit()

    def log_hit_status(self, mturk_hit_creation_response, task_group_id=None):
        """
        Create or update an entry in the hit status table.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        hit_details = mturk_hit_creation_response['HIT']
        id = hit_details['HITId']
        expiration = time.mktime(hit_details['Expiration'].timetuple())
        status = hit_details['HITStatus']
        assignments_pending = hit_details['NumberOfAssignmentsPending']
        assignments_available = hit_details['NumberOfAssignmentsAvailable']
        assignments_complete = hit_details['NumberOfAssignmentsCompleted']
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM hits WHERE hit_id = ?;', (id,))
            is_new_hit = c.fetchone()[0] == 0
            if is_new_hit:
                c.execute('UPDATE runs SET created = created + 1\n                             WHERE run_id = ?;', (task_group_id,))
            c.execute('REPLACE INTO hits VALUES (?,?,?,?,?,?,?);', (id, expiration, status, assignments_pending, assignments_available, assignments_complete, task_group_id))
            conn.commit()

    def log_worker_accept_assignment(self, worker_id, assignment_id, hit_id, task_group_id=None):
        """
        Log a worker accept, update assignment state and pairings to match the
        acceptance.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM workers WHERE worker_id = ?;', (worker_id,))
            has_worker = c.fetchone()[0] > 0
            if not has_worker:
                c.execute('INSERT INTO workers VALUES (?,?,?,?,?,?,?);', (worker_id, 1, 0, 0, 0, 0, 0))
            else:
                c.execute('UPDATE workers SET accepted = accepted + 1\n                             WHERE worker_id = ?;', (worker_id,))
            c.execute('REPLACE INTO assignments VALUES (?,?,?,?,?)', (assignment_id, 'Accepted', None, worker_id, hit_id))
            c.execute('INSERT INTO pairings\n                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (AssignState.STATUS_NONE, None, None, None, None, None, 0, '', False, '', worker_id, assignment_id, task_group_id, None, 0, ''))
            conn.commit()

    def log_complete_assignment(self, worker_id, assignment_id, approve_time, complete_type, task_group_id=None):
        """
        Note that an assignment was completed.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?, approve_time = ?\n                         WHERE assignment_id = ?;', ('Completed', approve_time, assignment_id))
            c.execute('UPDATE workers SET completed = completed + 1\n                         WHERE worker_id = ?;', (worker_id,))
            c.execute('UPDATE pairings SET status = ?, task_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (complete_type, time.time(), worker_id, assignment_id))
            c.execute('UPDATE runs SET completed = completed + 1\n                         WHERE run_id = ?;', (task_group_id,))
            conn.commit()

    def log_disconnect_assignment(self, worker_id, assignment_id, approve_time, disconnect_type, task_group_id=None):
        """
        Note that an assignment was disconnected from.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?, approve_time = ?\n                         WHERE assignment_id = ?;', ('Disconnected', approve_time, assignment_id))
            c.execute('UPDATE workers SET disconnected = disconnected + 1\n                         WHERE worker_id = ?;', (worker_id,))
            c.execute('UPDATE pairings SET status = ?, task_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (disconnect_type, time.time(), worker_id, assignment_id))
            c.execute('UPDATE runs SET failed = failed + 1 WHERE run_id = ?;', (task_group_id,))
            conn.commit()

    def log_expire_assignment(self, worker_id, assignment_id, task_group_id=None):
        """
        Note that an assignment was expired by us.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Expired', assignment_id))
            c.execute('UPDATE workers SET expired = expired + 1\n                         WHERE worker_id = ?;', (worker_id,))
            c.execute('UPDATE pairings SET status = ?, task_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (AssignState.STATUS_EXPIRED, time.time(), worker_id, assignment_id))
            c.execute('UPDATE runs SET failed = failed + 1 WHERE run_id = ?;', (task_group_id,))
            conn.commit()

    def log_submit_assignment(self, worker_id, assignment_id):
        """
        To be called whenever a worker hits the "submit hit" button.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Reviewable', assignment_id))
            conn.commit()

    def log_abandon_assignment(self, worker_id, assignment_id):
        """
        To be called whenever a worker returns a hit.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Abandoned', assignment_id))
            c.execute('UPDATE workers SET disconnected = disconnected + 1\n                         WHERE worker_id = ?;', (worker_id,))
            conn.commit()

    def log_start_onboard(self, worker_id, assignment_id, conversation_id):
        """
        Update a pairing state to reflect onboarding status.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE pairings SET status = ?, onboarding_start = ?,\n                         onboarding_id = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (AssignState.STATUS_ONBOARDING, time.time(), conversation_id, worker_id, assignment_id))
            conn.commit()

    def log_finish_onboard(self, worker_id, assignment_id):
        """
        Update a pairing state to reflect waiting status.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE pairings SET status = ?, onboarding_end = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (AssignState.STATUS_WAITING, time.time(), worker_id, assignment_id))
            conn.commit()

    def log_start_task(self, worker_id, assignment_id, conversation_id):
        """
        Update a pairing state to reflect in_task status.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE pairings SET status = ?, task_start = ?,\n                         conversation_id = ? WHERE worker_id = ?\n                         AND assignment_id = ?;', (AssignState.STATUS_IN_TASK, time.time(), conversation_id, worker_id, assignment_id))
            conn.commit()

    def log_award_amount(self, worker_id, assignment_id, amount, reason):
        """
        Update a pairing state to add a task bonus to be paid, appends reason.

        To be used for automatic evaluation bonuses
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            reason = '${} for {}\n'.format(amount, reason)
            cent_amount = int(amount * 100)
            c.execute('UPDATE pairings SET bonus_amount = bonus_amount + ?,\n                        bonus_text = bonus_text || ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (cent_amount, reason, worker_id, assignment_id))
            conn.commit()

    def log_pay_extra_bonus(self, worker_id, assignment_id, amount, reason):
        """
        Update a pairing state to add a bonus to be paid, appends reason.

        To be used for extra bonuses awarded at the discretion of the requester
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            reason = '${} for {}\n'.format(amount, reason)
            cent_amount = int(amount * 100)
            c.execute('UPDATE pairings\n                        SET extra_bonus_amount = extra_bonus_amount + ?,\n                        extra_bonus_text = extra_bonus_text || ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (cent_amount, reason, worker_id, assignment_id))
            conn.commit()

    def log_bonus_paid(self, worker_id, assignment_id):
        """
        Update to show that the intended bonus amount awarded for work in the task has
        been paid.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE pairings SET bonus_paid = ?\n                         WHERE worker_id = ? AND assignment_id = ?;', (True, worker_id, assignment_id))
            conn.commit()

    def log_approve_assignment(self, assignment_id):
        """
        Update assignment state to reflect approval, update worker state to increment
        number of accepted assignments.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
            assignment = c.fetchone()
            if assignment is None:
                return
            status = assignment['status']
            worker_id = assignment['worker_id']
            if status == 'Approved':
                return
            c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ?;', ('Approved', assignment_id))
            if status == 'Rejected':
                c.execute('UPDATE workers SET approved = approved + 1,\n                             rejected = rejected - 1\n                             WHERE worker_id = ?;', (worker_id,))
            else:
                c.execute('UPDATE workers SET approved = approved + 1\n                             WHERE worker_id = ?;', (worker_id,))
            conn.commit()

    def log_reject_assignment(self, assignment_id):
        """
        Update assignment state to reflect rejection, update worker state to increment
        number of rejected assignments.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('UPDATE assignments SET status = ?\n                         WHERE assignment_id = ? AND status != ?;', ('Rejected', assignment_id, 'Rejected'))
            if c.rowcount > 0:
                c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
                assignment = c.fetchone()
                if assignment is None:
                    return
                worker_id = assignment['worker_id']
                c.execute('UPDATE workers SET rejected = rejected + 1\n                             WHERE worker_id = ?;', (worker_id,))
            conn.commit()

    def log_worker_note(self, worker_id, assignment_id, note):
        """
        Append a note to the worker notes for a particular worker-assignment pairing.

        Adds newline to the note.
        """
        note += '\n'
        with self.table_access_condition:
            try:
                conn = self._get_connection()
                c = conn.cursor()
                c.execute('UPDATE pairings SET notes = notes || ?\n                             WHERE worker_id = ? AND assignment_id = ?;', (note, worker_id, assignment_id))
                conn.commit()
            except Exception as e:
                print(repr(e))

    def get_all_worker_data(self, start=0, count=100):
        """
        get all the worker data for all worker_ids.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM workers LIMIT ?,?;', (start, start + count))
            results = c.fetchall()
            return results

    def get_worker_data(self, worker_id):
        """
        get all worker data for a particular worker_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM workers WHERE worker_id = ?;', (worker_id,))
            results = c.fetchone()
            return results

    def get_assignments_for_run(self, task_group_id):
        """
        get all assignments for a particular run by task_group_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT assignments.* FROM assignments\n                         WHERE assignments.hit_id IN (\n                           SELECT hits.hit_id FROM hits\n                           WHERE hits.run_id = ?\n                         );', (task_group_id,))
            results = c.fetchall()
            return results

    def get_assignment_data(self, assignment_id):
        """
        get assignment data for a particular assignment by assignment_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM assignments WHERE assignment_id = ?;', (assignment_id,))
            results = c.fetchone()
            return results

    def get_worker_assignment_pairing(self, worker_id, assignment_id):
        """
        get a pairing data structure between a worker and an assignment.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE worker_id = ?\n                         AND assignment_id = ?;', (worker_id, assignment_id))
            results = c.fetchone()
            return results

    def get_all_run_data(self, start=0, count=1000):
        """
        get all the run data for all task_group_ids.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM runs LIMIT ?,?;', (start, start + count))
            results = c.fetchall()
            return results

    def get_run_data(self, task_group_id):
        """
        get the run data for the given task_group_id, return None if not found.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM runs WHERE run_id = ?;', (task_group_id,))
            results = c.fetchone()
            return results

    def get_hits_for_run(self, run_id):
        """
        Get the full list of HITs for the given run_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM hits WHERE run_id = ?;', (run_id,))
            results = c.fetchall()
            return results

    def get_hit_data(self, hit_id):
        """
        get the hit data for the given hit_id, return None if not.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM hits WHERE hit_id = ?;', (hit_id,))
            results = c.fetchone()
            return results

    def get_pairings_for_assignment(self, assignment_id):
        """
        get all pairings attached to a particular assignment_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE assignment_id = ?;', (assignment_id,))
            results = c.fetchall()
            return results

    def get_pairings_for_run(self, task_group_id):
        """
        get all pairings from a particular run by task_group_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings\n                         WHERE run_id = ?;', (task_group_id,))
            results = c.fetchall()
            return results

    def get_pairings_for_conversation(self, conversation_id, task_group_id=None):
        """
        get all pairings for a singular conversation in a run by conversation_id and
        task_group_id.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE conversation_id = ?\n                         AND run_id = ?;', (conversation_id, task_group_id))
            results = c.fetchall()
            return results

    def get_all_assignments_for_worker(self, worker_id):
        """
        get all assignments associated with a particular worker_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM assignments WHERE worker_id = ?;', (worker_id,))
            results = c.fetchall()
            return results

    def get_all_pairings_for_worker(self, worker_id):
        """
        get all pairings associated with a particular worker_id.
        """
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE worker_id = ?;', (worker_id,))
            results = c.fetchall()
            return results

    def get_all_task_assignments_for_worker(self, worker_id, task_group_id=None):
        """
        get all assignments for a particular worker within a particular run by worker_id
        and task_group_id.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT assignments.assignment_id, assignments.status,\n                         assignments.approve_time, assignments.worker_id,\n                         assignments.hit_id\n                         FROM assignments\n                         INNER JOIN hits on assignments.hit_id = hits.hit_id\n                         WHERE assignments.worker_id = ? AND hits.run_id = ?;\n                         ', (worker_id, task_group_id))
            results = c.fetchall()
            return results

    def get_all_task_pairings_for_worker(self, worker_id, task_group_id=None):
        """
        get all pairings for a particular worker within a particular run by worker_id
        and task_group_id.
        """
        task_group_id = self._force_task_group_id(task_group_id)
        with self.table_access_condition:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM pairings WHERE worker_id = ?\n                         AND run_id = ?;', (worker_id, task_group_id))
            results = c.fetchall()
            return results

    @staticmethod
    def get_conversation_data(task_group_id, conv_id, worker_id, is_sandbox):
        """
        A poorly named function that gets conversation data for a worker.
        """
        result = {'had_data_dir': False, 'had_run_dir': False, 'had_conversation_dir': False, 'had_worker_dir': False, 'had_worker_file': False, 'data': None}
        target = 'sandbox' if is_sandbox else 'live'
        search_dir = os.path.join(data_dir, target)
        if not os.path.exists(search_dir):
            return result
        result['had_data_dir'] = True
        search_dir = os.path.join(search_dir, task_group_id)
        if not os.path.exists(search_dir):
            return result
        result['had_run_dir'] = True
        search_dir = os.path.join(search_dir, conv_id)
        if not os.path.exists(search_dir):
            return result
        result['had_conversation_dir'] = True
        search_dir = os.path.join(search_dir, 'workers')
        if not os.path.exists(search_dir):
            return result
        result['had_worker_dir'] = True
        target_filename = os.path.join(search_dir, '{}.json'.format(worker_id))
        if not os.path.exists(target_filename):
            return result
        result['had_worker_file'] = True
        with open(target_filename, 'r') as target_file:
            result['data'] = json.load(target_file)
        return result

    @staticmethod
    def get_full_conversation_data(task_group_id, conv_id, is_sandbox):
        """
        Gets all conversation data saved for a world.
        """
        target = 'sandbox' if is_sandbox else 'live'
        return_data = {'custom_data': {}, 'worker_data': {}}
        target_dir = os.path.join(data_dir, target, task_group_id, conv_id)
        target_dir_custom = os.path.join(target_dir, 'custom')
        custom_file = os.path.join(target_dir_custom, 'data.json')
        if os.path.exists(custom_file):
            custom_data = {}
            with open(custom_file, 'r') as infile:
                custom_data = json.load(infile)
            pickle_file = os.path.join(target_dir_custom, 'data.pickle')
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as infile:
                    custom_data['needs-pickle'] = pickle.load(infile)
            return_data['custom_data'] = custom_data
        target_dir_workers = os.path.join(target_dir, 'workers')
        for w_file in os.listdir(target_dir_workers):
            w_id = w_file.split('.json')[0]
            worker_file = os.path.join(target_dir_workers, w_file)
            with open(worker_file, 'r') as infile:
                return_data['worker_data'][w_id] = json.load(infile)
        return return_data