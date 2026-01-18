from .dom_helpers import try_urljoin
from .mf2_classes import filter_classes
Extracts and returns a metaformats item from a BeautifulSoup parse tree.

    Args:
      soup (bs4.BeautifulSoup): parsed HTML
      url (str): URL of document

    Returns:
      dict: mf2 item, or None if the input is not eligible for metaformats
    