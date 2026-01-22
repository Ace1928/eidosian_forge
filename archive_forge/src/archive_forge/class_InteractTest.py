from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import interact
from fire import testutils
import mock
class InteractTest(testutils.BaseTestCase):

    @mock.patch(INTERACT_METHOD)
    def testInteract(self, mock_interact_method):
        self.assertFalse(mock_interact_method.called)
        interact.Embed({})
        self.assertTrue(mock_interact_method.called)

    @mock.patch(INTERACT_METHOD)
    def testInteractVariables(self, mock_interact_method):
        self.assertFalse(mock_interact_method.called)
        interact.Embed({'count': 10, 'mock': mock})
        self.assertTrue(mock_interact_method.called)