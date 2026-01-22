import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
def _update_password_created_at(self, user_id, password_create_at):
    with sql.session_for_write() as session:
        user_ref = session.get(model.User, user_id)
        latest_password = user_ref.password_ref
        slightly_less = datetime.timedelta(minutes=1)
        for password_ref in user_ref.local_user.passwords:
            password_ref.created_at = password_create_at - slightly_less
        latest_password.created_at = password_create_at