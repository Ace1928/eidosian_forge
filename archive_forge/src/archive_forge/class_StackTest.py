from unittest import mock
from oslo_utils import timeutils
from heat.common import timeutils as heat_timeutils
from heat.engine import notification
from heat.tests import common
from heat.tests import utils
class StackTest(common.HeatTestCase):

    def setUp(self):
        super(StackTest, self).setUp()
        self.ctx = utils.dummy_context(user_id='test_user_id')

    def test_send(self):
        created_time = timeutils.utcnow()
        st = mock.Mock()
        st.state = ('x', 'f')
        st.status = st.state[0]
        st.action = st.state[1]
        st.name = 'fred'
        st.status_reason = 'this is why'
        st.created_time = created_time
        st.context = self.ctx
        st.id = 'hay-are-en'
        updated_time = timeutils.utcnow()
        st.updated_time = updated_time
        st.tags = ['tag1', 'tag2']
        st.t = mock.MagicMock()
        st.t.__getitem__.return_value = 'for test'
        st.t.DESCRIPTION = 'description'
        notify = self.patchobject(notification, 'notify')
        notification.stack.send(st)
        notify.assert_called_once_with(self.ctx, 'stack.f.error', 'ERROR', {'state_reason': 'this is why', 'user_id': 'test_username', 'username': 'test_username', 'user_identity': 'test_user_id', 'stack_identity': 'hay-are-en', 'stack_name': 'fred', 'tenant_id': 'test_tenant_id', 'create_at': heat_timeutils.isotime(created_time), 'state': 'x_f', 'description': 'for test', 'tags': ['tag1', 'tag2'], 'updated_at': heat_timeutils.isotime(updated_time)})