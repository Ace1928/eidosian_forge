from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.utils import translation
def _urls(self, page, protocol, domain):
    urls = []
    latest_lastmod = None
    all_items_lastmod = True
    paginator_page = self.paginator.page(page)
    for item in paginator_page.object_list:
        loc = f'{protocol}://{domain}{self._location(item)}'
        priority = self._get('priority', item)
        lastmod = self._get('lastmod', item)
        if all_items_lastmod:
            all_items_lastmod = lastmod is not None
            if all_items_lastmod and (latest_lastmod is None or lastmod > latest_lastmod):
                latest_lastmod = lastmod
        url_info = {'item': item, 'location': loc, 'lastmod': lastmod, 'changefreq': self._get('changefreq', item), 'priority': str(priority if priority is not None else ''), 'alternates': []}
        if self.i18n and self.alternates:
            item_languages = self.get_languages_for_item(item[0])
            for lang_code in item_languages:
                loc = f'{protocol}://{domain}{self._location(item, lang_code)}'
                url_info['alternates'].append({'location': loc, 'lang_code': lang_code})
            if self.x_default and settings.LANGUAGE_CODE in item_languages:
                lang_code = settings.LANGUAGE_CODE
                loc = f'{protocol}://{domain}{self._location(item, lang_code)}'
                loc = loc.replace(f'/{lang_code}/', '/', 1)
                url_info['alternates'].append({'location': loc, 'lang_code': 'x-default'})
        urls.append(url_info)
    if all_items_lastmod and latest_lastmod:
        self.latest_lastmod = latest_lastmod
    return urls