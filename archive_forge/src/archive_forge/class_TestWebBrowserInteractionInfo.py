from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.bakery as bakery
class TestWebBrowserInteractionInfo(TestCase):

    def test_from_dict(self):
        info_dict = {'VisitURL': 'https://example.com/visit', 'WaitTokenURL': 'https://example.com/wait'}
        interaction_info = httpbakery.WebBrowserInteractionInfo.from_dict(info_dict)
        self.assertEqual(interaction_info.visit_url, 'https://example.com/visit')
        self.assertEqual(interaction_info.wait_token_url, 'https://example.com/wait')