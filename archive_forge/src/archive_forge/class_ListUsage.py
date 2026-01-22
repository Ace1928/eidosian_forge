import datetime
import functools
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListUsage(command.Lister):
    _description = _('List resource usage per project')

    def get_parser(self, prog_name):
        parser = super(ListUsage, self).get_parser(prog_name)
        parser.add_argument('--start', metavar='<start>', default=None, help=_('Usage range start date, ex 2012-01-20 (default: 4 weeks ago)'))
        parser.add_argument('--end', metavar='<end>', default=None, help=_('Usage range end date, ex 2012-01-20 (default: tomorrow)'))
        return parser

    def take_action(self, parsed_args):

        def _format_project(project):
            if not project:
                return ''
            if project in project_cache.keys():
                return project_cache[project].name
            else:
                return project
        compute_client = self.app.client_manager.sdk_connection.compute
        columns = ('project_id', 'server_usages', 'total_memory_mb_usage', 'total_vcpus_usage', 'total_local_gb_usage')
        column_headers = ('Project', 'Servers', 'RAM MB-Hours', 'CPU Hours', 'Disk GB-Hours')
        date_cli_format = '%Y-%m-%d'
        now = datetime.datetime.utcnow()
        if parsed_args.start:
            start = datetime.datetime.strptime(parsed_args.start, date_cli_format)
        else:
            start = now - datetime.timedelta(weeks=4)
        if parsed_args.end:
            end = datetime.datetime.strptime(parsed_args.end, date_cli_format)
        else:
            end = now + datetime.timedelta(days=1)
        usage_list = list(compute_client.usages(start=start, end=end, detailed=True))
        project_cache = {}
        try:
            for p in self.app.client_manager.identity.projects.list():
                project_cache[p.id] = p
        except Exception:
            pass
        if parsed_args.formatter == 'table' and len(usage_list) > 0:
            self.app.stdout.write(_('Usage from %(start)s to %(end)s: \n') % {'start': start.strftime(date_cli_format), 'end': end.strftime(date_cli_format)})
        return (column_headers, (utils.get_item_properties(s, columns, formatters=_formatters(project_cache)) for s in usage_list))