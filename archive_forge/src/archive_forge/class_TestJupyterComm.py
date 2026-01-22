from pyviz_comms import Comm, JupyterComm
from holoviews.element.comparison import ComparisonTestCase
class TestJupyterComm(ComparisonTestCase):

    def test_init_comm(self):
        JupyterComm()

    def test_init_comm_id(self):
        comm = JupyterComm(id='Test')
        self.assertEqual(comm.id, 'Test')

    def test_decode(self):
        msg = {'content': {'data': 'Test'}}
        decoded = JupyterComm.decode(msg)
        self.assertEqual(decoded, 'Test')