from keystone.i18n import _
def _build_user(self, user_id=None, username=None, user_domain_id=None, user_domain_name=None):
    user = {}
    if user_id:
        user['id'] = user_id
    else:
        user['name'] = username
        if user_domain_id or user_domain_name:
            user['domain'] = {}
            if user_domain_id:
                user['domain']['id'] = user_domain_id
            else:
                user['domain']['name'] = user_domain_name
    return user