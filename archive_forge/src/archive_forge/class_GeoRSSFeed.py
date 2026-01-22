from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
class GeoRSSFeed(Rss201rev2Feed, GeoFeedMixin):

    def rss_attributes(self):
        attrs = super().rss_attributes()
        attrs['xmlns:georss'] = 'http://www.georss.org/georss'
        return attrs

    def add_item_elements(self, handler, item):
        super().add_item_elements(handler, item)
        self.add_georss_element(handler, item)

    def add_root_elements(self, handler):
        super().add_root_elements(handler)
        self.add_georss_element(handler, self.feed)