import unittest
import os
import time
import uuid
from datetime import datetime
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.mturk_data_handler as DataHandlerFile
def assertHITEqual(self, mturk_hit, db_hit, run_id):
    self.assertEqual(mturk_hit['HIT']['HITId'], db_hit['hit_id'])
    self.assertEqual(mturk_hit['HIT']['Expiration'], datetime.fromtimestamp(db_hit['expiration']))
    self.assertEqual(mturk_hit['HIT']['HITStatus'], db_hit['hit_status'])
    self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsPending'], db_hit['assignments_pending'])
    self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsAvailable'], db_hit['assignments_available'])
    self.assertEqual(mturk_hit['HIT']['NumberOfAssignmentsCompleted'], db_hit['assignments_complete'])
    self.assertEqual(run_id, db_hit['run_id'])