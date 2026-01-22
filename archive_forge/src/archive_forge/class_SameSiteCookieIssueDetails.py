from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import page
@dataclass
class SameSiteCookieIssueDetails:
    """
    This information is currently necessary, as the front-end has a difficult
    time finding a specific cookie. With this, we can convey specific error
    information without the cookie.
    """
    cookie: AffectedCookie
    cookie_warning_reasons: typing.List[SameSiteCookieWarningReason]
    cookie_exclusion_reasons: typing.List[SameSiteCookieExclusionReason]
    operation: SameSiteCookieOperation
    site_for_cookies: typing.Optional[str] = None
    cookie_url: typing.Optional[str] = None
    request: typing.Optional[AffectedRequest] = None

    def to_json(self):
        json = dict()
        json['cookie'] = self.cookie.to_json()
        json['cookieWarningReasons'] = [i.to_json() for i in self.cookie_warning_reasons]
        json['cookieExclusionReasons'] = [i.to_json() for i in self.cookie_exclusion_reasons]
        json['operation'] = self.operation.to_json()
        if self.site_for_cookies is not None:
            json['siteForCookies'] = self.site_for_cookies
        if self.cookie_url is not None:
            json['cookieUrl'] = self.cookie_url
        if self.request is not None:
            json['request'] = self.request.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(cookie=AffectedCookie.from_json(json['cookie']), cookie_warning_reasons=[SameSiteCookieWarningReason.from_json(i) for i in json['cookieWarningReasons']], cookie_exclusion_reasons=[SameSiteCookieExclusionReason.from_json(i) for i in json['cookieExclusionReasons']], operation=SameSiteCookieOperation.from_json(json['operation']), site_for_cookies=str(json['siteForCookies']) if 'siteForCookies' in json else None, cookie_url=str(json['cookieUrl']) if 'cookieUrl' in json else None, request=AffectedRequest.from_json(json['request']) if 'request' in json else None)