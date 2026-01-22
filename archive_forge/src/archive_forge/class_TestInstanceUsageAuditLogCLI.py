import datetime
from oslo_utils import timeutils
from novaclient.tests.functional import base
class TestInstanceUsageAuditLogCLI(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.1'

    @staticmethod
    def _get_begin_end_time():
        current = timeutils.utcnow()
        end = datetime.datetime(day=1, month=current.month, year=current.year)
        year = end.year
        if current.month == 1:
            year -= 1
            month = 12
        else:
            month = current.month - 1
        begin = datetime.datetime(day=1, month=month, year=year)
        return (begin, end)

    def test_get_os_instance_usage_audit_log(self):
        begin, end = self._get_begin_end_time()
        expected = {'hosts_not_run': '[]', 'log': '{}', 'num_hosts': '0', 'num_hosts_done': '0', 'num_hosts_not_run': '0', 'num_hosts_running': '0', 'overall_status': 'ALL hosts done. 0 errors.', 'total_errors': '0', 'total_instances': '0', 'period_beginning': str(begin), 'period_ending': str(end)}
        output = self.nova('instance-usage-audit-log')
        for key in expected.keys():
            self.assertEqual(expected[key], self._get_value_from_the_table(output, key))

    def test_get_os_instance_usage_audit_log_with_before(self):
        expected = {'hosts_not_run': '[]', 'log': '{}', 'num_hosts': '0', 'num_hosts_done': '0', 'num_hosts_not_run': '0', 'num_hosts_running': '0', 'overall_status': 'ALL hosts done. 0 errors.', 'total_errors': '0', 'total_instances': '0', 'period_beginning': '2016-11-01 00:00:00', 'period_ending': '2016-12-01 00:00:00'}
        output = self.nova('instance-usage-audit-log --before "2016-12-10 13:59:59.999999"')
        for key in expected.keys():
            self.assertEqual(expected[key], self._get_value_from_the_table(output, key))