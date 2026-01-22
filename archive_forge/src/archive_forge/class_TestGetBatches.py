import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class TestGetBatches(common.HeatTestCase):
    scenarios = [('4_4_1_0', dict(tgt_cap=4, curr_cap=4, bat_size=1, min_serv=0, batches=[(4, 1)] * 4)), ('3_4_1_0', dict(tgt_cap=3, curr_cap=4, bat_size=1, min_serv=0, batches=[(3, 1)] * 3)), ('4_4_1_4', dict(tgt_cap=4, curr_cap=4, bat_size=1, min_serv=4, batches=[(5, 1)] * 4 + [(4, 0)])), ('4_4_1_5', dict(tgt_cap=4, curr_cap=4, bat_size=1, min_serv=5, batches=[(5, 1)] * 4 + [(4, 0)])), ('4_4_2_0', dict(tgt_cap=4, curr_cap=4, bat_size=2, min_serv=0, batches=[(4, 2)] * 2)), ('4_4_2_4', dict(tgt_cap=4, curr_cap=4, bat_size=2, min_serv=4, batches=[(6, 2)] * 2 + [(4, 0)])), ('5_5_2_0', dict(tgt_cap=5, curr_cap=5, bat_size=2, min_serv=0, batches=[(5, 2)] * 2 + [(5, 1)])), ('5_5_2_4', dict(tgt_cap=5, curr_cap=5, bat_size=2, min_serv=4, batches=[(6, 2)] * 2 + [(5, 1)])), ('3_3_2_0', dict(tgt_cap=3, curr_cap=3, bat_size=2, min_serv=0, batches=[(3, 2), (3, 1)])), ('3_3_2_4', dict(tgt_cap=3, curr_cap=3, bat_size=2, min_serv=4, batches=[(5, 2), (4, 1), (3, 0)])), ('4_4_4_0', dict(tgt_cap=4, curr_cap=4, bat_size=4, min_serv=0, batches=[(4, 4)])), ('4_4_5_0', dict(tgt_cap=4, curr_cap=4, bat_size=5, min_serv=0, batches=[(4, 4)])), ('4_4_4_1', dict(tgt_cap=4, curr_cap=4, bat_size=4, min_serv=1, batches=[(5, 4), (4, 0)])), ('4_4_6_1', dict(tgt_cap=4, curr_cap=4, bat_size=6, min_serv=1, batches=[(5, 4), (4, 0)])), ('4_4_4_2', dict(tgt_cap=4, curr_cap=4, bat_size=4, min_serv=2, batches=[(6, 4), (4, 0)])), ('4_4_4_4', dict(tgt_cap=4, curr_cap=4, bat_size=4, min_serv=4, batches=[(8, 4), (4, 0)])), ('4_4_5_6', dict(tgt_cap=4, curr_cap=4, bat_size=5, min_serv=6, batches=[(8, 4), (4, 0)]))]

    def test_get_batches(self):
        batches = list(instgrp.InstanceGroup._get_batches(self.tgt_cap, self.curr_cap, self.bat_size, self.min_serv))
        self.assertEqual(self.batches, batches)