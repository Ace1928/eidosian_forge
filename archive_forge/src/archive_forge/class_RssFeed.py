import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class RssFeed(SyndicationFeed):
    content_type = 'application/rss+xml; charset=utf-8'

    def write(self, outfile, encoding):
        handler = SimplerXMLGenerator(outfile, encoding, short_empty_elements=True)
        handler.startDocument()
        handler.startElement('rss', self.rss_attributes())
        handler.startElement('channel', self.root_attributes())
        self.add_root_elements(handler)
        self.write_items(handler)
        self.endChannelElement(handler)
        handler.endElement('rss')

    def rss_attributes(self):
        return {'version': self._version, 'xmlns:atom': 'http://www.w3.org/2005/Atom'}

    def write_items(self, handler):
        for item in self.items:
            handler.startElement('item', self.item_attributes(item))
            self.add_item_elements(handler, item)
            handler.endElement('item')

    def add_root_elements(self, handler):
        handler.addQuickElement('title', self.feed['title'])
        handler.addQuickElement('link', self.feed['link'])
        handler.addQuickElement('description', self.feed['description'])
        if self.feed['feed_url'] is not None:
            handler.addQuickElement('atom:link', None, {'rel': 'self', 'href': self.feed['feed_url']})
        if self.feed['language'] is not None:
            handler.addQuickElement('language', self.feed['language'])
        for cat in self.feed['categories']:
            handler.addQuickElement('category', cat)
        if self.feed['feed_copyright'] is not None:
            handler.addQuickElement('copyright', self.feed['feed_copyright'])
        handler.addQuickElement('lastBuildDate', rfc2822_date(self.latest_post_date()))
        if self.feed['ttl'] is not None:
            handler.addQuickElement('ttl', self.feed['ttl'])

    def endChannelElement(self, handler):
        handler.endElement('channel')