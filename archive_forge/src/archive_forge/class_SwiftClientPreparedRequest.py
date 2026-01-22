import requests
from requests.sessions import merge_setting, merge_hooks
from requests.structures import CaseInsensitiveDict
class SwiftClientPreparedRequest(requests.PreparedRequest):

    def prepare_headers(self, headers):
        try:
            return super().prepare_headers(headers)
        except UnicodeError:
            self.headers = CaseInsensitiveDict(headers or {})