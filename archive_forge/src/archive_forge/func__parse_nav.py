import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def _parse_nav(self, data, base_path, navtype='toc'):
    html_node = parse_html_string(data)
    if navtype == 'toc':
        nav_node = html_node.xpath("//nav[@*='toc']")[0]
    else:
        _page_list = html_node.xpath("//nav[@*='page-list']")
        if len(_page_list) == 0:
            return
        nav_node = _page_list[0]

    def parse_list(list_node):
        items = []
        for item_node in list_node.findall('li'):
            sublist_node = item_node.find('ol')
            link_node = item_node.find('a')
            if sublist_node is not None:
                title = item_node[0].text
                children = parse_list(sublist_node)
                if link_node is not None:
                    href = zip_path.normpath(zip_path.join(base_path, link_node.get('href')))
                    items.append((Section(title, href=href), children))
                else:
                    items.append((Section(title), children))
            elif link_node is not None:
                title = link_node.text
                href = zip_path.normpath(zip_path.join(base_path, link_node.get('href')))
                items.append(Link(href, title))
        return items
    if navtype == 'toc':
        self.book.toc = parse_list(nav_node.find('ol'))
    elif nav_node is not None:
        self.book.pages = parse_list(nav_node.find('ol'))
        htmlfiles = dict()
        for htmlfile in self.book.items:
            if isinstance(htmlfile, EpubHtml):
                htmlfiles[htmlfile.file_name] = htmlfile
        for page in self.book.pages:
            try:
                filename, idref = page.href.split('#')
            except ValueError:
                filename = page.href
            if filename in htmlfiles:
                htmlfiles[filename].pages.append(page)