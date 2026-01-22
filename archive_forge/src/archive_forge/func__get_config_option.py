from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.resource.config_backends import base
def _get_config_option(self, session, domain_id, group, option, sensitive):
    try:
        config_table = self.choose_table(sensitive)
        ref = session.query(config_table).filter_by(domain_id=domain_id, group=group, option=option).one()
    except sql.NotFound:
        msg = _('option %(option)s in group %(group)s') % {'group': group, 'option': option}
        raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
    return ref