from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import exceptions
class MemberFormatter(base.MistralFormatter):
    COLUMNS = [('resource_id', 'Resource ID'), ('resource_type', 'Resource Type'), ('project_id', 'Resource Owner'), ('member_id', 'Member ID'), ('status', 'Status'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(member=None, lister=False):
        if member:
            data = (member.resource_id, member.resource_type, member.project_id, member.member_id, member.status, member.created_at)
            if hasattr(member, 'updated_at'):
                data += (member.updated_at,)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(MemberFormatter.COLUMNS)))),)
        return (MemberFormatter.headings(), data)