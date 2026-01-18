import sys
from typing import Iterator, Optional
from urllib.parse import ParseResult, urlparse
from .config import ConfigDict, SectionLike
def urlmatch_credential_sections(config: ConfigDict, url: Optional[str]) -> Iterator[SectionLike]:
    """Returns credential sections from the config which match the given URL."""
    encoding = config.encoding or sys.getdefaultencoding()
    parsed_url = urlparse(url or '')
    for config_section in config.sections():
        if config_section[0] != b'credential':
            continue
        if len(config_section) < 2:
            yield config_section
            continue
        config_url = config_section[1].decode(encoding)
        parsed_config_url = urlparse(config_url)
        if parsed_config_url.scheme and parsed_config_url.netloc:
            is_match = match_urls(parsed_url, parsed_config_url)
        else:
            is_match = match_partial_url(parsed_url, config_url)
        if is_match:
            yield config_section