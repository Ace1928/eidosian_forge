import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
def _process_reply(self, msg):
    if msg.type == ovs.jsonrpc.Message.T_ERROR:
        self._status = Transaction.ERROR
    elif not isinstance(msg.result, (list, tuple)):
        vlog.warn('reply to "transact" is not JSON array')
    else:
        hard_errors = False
        soft_errors = False
        lock_errors = False
        ops = msg.result
        for op in ops:
            if op is None:
                soft_errors = True
            elif isinstance(op, dict):
                error = op.get('error')
                if error is not None:
                    if error == 'timed out':
                        soft_errors = True
                    elif error == 'not owner':
                        lock_errors = True
                    elif error == 'aborted':
                        pass
                    else:
                        hard_errors = True
                        self.__set_error_json(op)
            else:
                hard_errors = True
                self.__set_error_json(op)
                vlog.warn('operation reply is not JSON null or object')
        if not soft_errors and (not hard_errors) and (not lock_errors):
            if self._inc_row and (not self.__process_inc_reply(ops)):
                hard_errors = True
            if self._fetch_requests:
                if self.__process_fetch_reply(ops):
                    self.idl.change_seqno += 1
                else:
                    hard_errors = True
            for insert in self._inserted_rows.values():
                if not self.__process_insert_reply(insert, ops):
                    hard_errors = True
        if hard_errors:
            self._status = Transaction.ERROR
        elif lock_errors:
            self._status = Transaction.NOT_LOCKED
        elif soft_errors:
            self._status = Transaction.TRY_AGAIN
        else:
            self._status = Transaction.SUCCESS