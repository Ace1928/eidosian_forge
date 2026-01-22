import re
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import mail_managers
from django.http import HttpResponsePermanentRedirect
from django.urls import is_valid_path
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import escape_leading_slashes
class BrokenLinkEmailsMiddleware(MiddlewareMixin):

    def process_response(self, request, response):
        """Send broken link emails for relevant 404 NOT FOUND responses."""
        if response.status_code == 404 and (not settings.DEBUG):
            domain = request.get_host()
            path = request.get_full_path()
            referer = request.META.get('HTTP_REFERER', '')
            if not self.is_ignorable_request(request, path, domain, referer):
                ua = request.META.get('HTTP_USER_AGENT', '<none>')
                ip = request.META.get('REMOTE_ADDR', '<none>')
                mail_managers('Broken %slink on %s' % ('INTERNAL ' if self.is_internal_request(domain, referer) else '', domain), 'Referrer: %s\nRequested URL: %s\nUser agent: %s\nIP address: %s\n' % (referer, path, ua, ip), fail_silently=True)
        return response

    def is_internal_request(self, domain, referer):
        """
        Return True if the referring URL is the same domain as the current
        request.
        """
        return bool(re.match('^https?://%s/' % re.escape(domain), referer))

    def is_ignorable_request(self, request, uri, domain, referer):
        """
        Return True if the given request *shouldn't* notify the site managers
        according to project settings or in situations outlined by the inline
        comments.
        """
        if not referer:
            return True
        if settings.APPEND_SLASH and uri.endswith('/') and (referer == uri[:-1]):
            return True
        if not self.is_internal_request(domain, referer) and '?' in referer:
            return True
        parsed_referer = urlparse(referer)
        if parsed_referer.netloc in ['', domain] and parsed_referer.path == uri:
            return True
        return any((pattern.search(uri) for pattern in settings.IGNORABLE_404_URLS))