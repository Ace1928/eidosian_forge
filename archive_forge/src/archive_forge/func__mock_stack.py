from unittest import mock
from oslo_utils import timeutils
from heat.common import timeutils as heat_timeutils
from heat.engine import notification
from heat.tests import common
from heat.tests import utils
def _mock_stack(self):
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
    return st