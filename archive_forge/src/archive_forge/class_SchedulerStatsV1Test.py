from manilaclient.tests.unit import utils
class SchedulerStatsV1Test(utils.TestCase):

    def test_import_v1_scheduler_stats_module(self):
        try:
            from manilaclient.v1 import scheduler_stats
        except Exception as e:
            msg = "module 'manilaclient.v1.scheduler_stats' cannot be imported with error: %s" % str(e)
            assert False, msg
        for cls in ('Pool', 'PoolManager'):
            msg = "Module 'scheduler_stats' has no '%s' attr." % cls
            self.assertTrue(hasattr(scheduler_stats, cls), msg)