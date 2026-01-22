import re
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor as LinkExtractor
def _is_valid_url(url):
    return url.split('://', 1)[0] in {'http', 'https', 'file', 'ftp'}