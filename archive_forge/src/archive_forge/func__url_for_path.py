import urllib.parse
def _url_for_path(self, path: str) -> str:
    return urllib.parse.urljoin(self.url, path)