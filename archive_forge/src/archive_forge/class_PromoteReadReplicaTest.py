import unittest
import time
from boto.rds import RDSConnection
class PromoteReadReplicaTest(unittest.TestCase):
    rds = True

    def setUp(self):
        self.conn = RDSConnection()
        self.masterDB_name = 'boto-db-%s' % str(int(time.time()))
        self.replicaDB_name = 'replica-%s' % self.masterDB_name
        self.renamedDB_name = 'renamed-replica-%s' % self.masterDB_name

    def tearDown(self):
        instances = self.conn.get_all_dbinstances()
        for db in [self.masterDB_name, self.replicaDB_name, self.renamedDB_name]:
            for i in instances:
                if i.id == db:
                    self.conn.delete_dbinstance(db, skip_final_snapshot=True)

    def test_promote(self):
        print('--- running RDS promotion & renaming tests ---')
        self.masterDB = self.conn.create_dbinstance(self.masterDB_name, 5, 'db.t1.micro', 'root', 'bototestpw')
        print('--- waiting for "%s" to become available  ---' % self.masterDB_name)
        wait_timeout = time.time() + 15 * 60
        time.sleep(60)
        instances = self.conn.get_all_dbinstances(self.masterDB_name)
        inst = instances[0]
        while wait_timeout > time.time() and inst.status != 'available':
            time.sleep(15)
            instances = self.conn.get_all_dbinstances(self.masterDB_name)
            inst = instances[0]
        self.assertTrue(inst.status == 'available')
        self.replicaDB = self.conn.create_dbinstance_read_replica(self.replicaDB_name, self.masterDB_name)
        print('--- waiting for "%s" to become available  ---' % self.replicaDB_name)
        wait_timeout = time.time() + 15 * 60
        time.sleep(60)
        instances = self.conn.get_all_dbinstances(self.replicaDB_name)
        inst = instances[0]
        while wait_timeout > time.time() and inst.status != 'available':
            time.sleep(15)
            instances = self.conn.get_all_dbinstances(self.replicaDB_name)
            inst = instances[0]
        self.assertTrue(inst.status == 'available')
        self.replicaDB = self.conn.promote_read_replica(self.replicaDB_name)
        print('--- waiting for "%s" to be promoted and available  ---' % self.replicaDB_name)
        wait_timeout = time.time() + 15 * 60
        time.sleep(60)
        instances = self.conn.get_all_dbinstances(self.replicaDB_name)
        inst = instances[0]
        while wait_timeout > time.time() and inst.status != 'available':
            time.sleep(15)
            instances = self.conn.get_all_dbinstances(self.replicaDB_name)
            inst = instances[0]
        self.assertTrue(inst)
        self.assertTrue(inst.status == 'available')
        self.assertFalse(inst.status_infos)
        instances = self.conn.get_all_dbinstances(self.masterDB_name)
        inst = instances[0]
        self.assertFalse(inst.read_replica_dbinstance_identifiers)
        print('--- renaming "%s" to "%s" ---' % (self.replicaDB_name, self.renamedDB_name))
        self.renamedDB = self.conn.modify_dbinstance(self.replicaDB_name, new_instance_id=self.renamedDB_name, apply_immediately=True)
        print('--- waiting for "%s" to exist  ---' % self.renamedDB_name)
        wait_timeout = time.time() + 15 * 60
        time.sleep(60)
        found = False
        while found == False and wait_timeout > time.time():
            instances = self.conn.get_all_dbinstances()
            for i in instances:
                if i.id == self.renamedDB_name:
                    found = True
            if found == False:
                time.sleep(15)
        self.assertTrue(found)
        print('--- waiting for "%s" to become available ---' % self.renamedDB_name)
        instances = self.conn.get_all_dbinstances(self.renamedDB_name)
        inst = instances[0]
        while wait_timeout > time.time() and inst.status != 'available':
            time.sleep(15)
            instances = self.conn.get_all_dbinstances(self.renamedDB_name)
            inst = instances[0]
        self.assertTrue(inst.status == 'available')
        self.replicaDB = None
        print('--- tests completed ---')