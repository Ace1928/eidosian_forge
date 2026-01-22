from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
class URLMap(HandlerBase):
    """Maps from URLs to handlers.

  This class acts similar to a union type. Its purpose is to describe a mapping
  between a set of URLs and their handlers. The handler type of a given instance
  is determined by which `handler-id` attribute is used.

  Every mapping can have one and only one handler type. Attempting to use more
  than one `handler-id` attribute will cause an `UnknownHandlerType` to be
  raised during validation. Failure to provide any `handler-id` attributes will
  cause `MissingHandlerType` to be raised during validation.

  The regular expression used by the `url` field will be used to match against
  the entire URL path and query string of the request; therefore, partial maps
  will not be matched. Specifying a `url`, such as `/admin`, is the same as
  matching against the regular expression `^/admin$`. Don't start your matching
  `url` with `^` or end them with `$`. These regular expressions won't be
  accepted and will raise `ValueError`.

  Attributes:
    login: Specifies whether a user should be logged in to access a URL.
        The default value of this argument is `optional`.
    secure: Sets the restriction on the protocol that can be used to serve this
        URL or handler. This value can be set to `HTTP`, `HTTPS` or `either`.
    url: Specifies a regular expression that is used to fully match against the
        request URLs path. See the "Special cases" section of this document to
        learn more.
    static_files: Specifies the handler ID attribute that maps `url` to the
        appropriate file. You can specify regular expression backreferences to
        the string matched to `url`.
    upload: Specifies the regular expression that is used by the application
        configuration program to determine which files are uploaded as blobs.
        Because it is difficult to determine this information using just the
        `url` and `static_files` arguments, this attribute must be included.
        This attribute is required when you define a `static_files` mapping. A
        matching file name must fully match against the `upload` regular
        expression, similar to how `url` is matched against the request path. Do
        not begin the `upload` argument with the `^` character or end it with
        the `$` character.
    static_dir: Specifies the handler ID that maps the provided `url` to a
        sub-directory within the application directory. See "Special cases."
    mime_type: When used with `static_files` and `static_dir`, this argument
        specifies that the MIME type of the files that are served from those
        directories must be overridden with this value.
    script: Specifies the handler ID that maps URLs to a script handler within
        the application directory that will run using CGI.
    position: Used in `AppInclude` objects to specify whether a handler should
        be inserted at the beginning of the primary handler list or at the end.
        If `tail` is specified, the handler is inserted at the end; otherwise,
        the handler is inserted at the beginning. This behavior implies that
        `head` is the effective default.
    expiration: When used with static files and directories, this argument
        specifies the time delta to use for cache expiration. This argument
        should use the following format: `4d 5h 30m 15s`, where each letter
        signifies days, hours, minutes, and seconds, respectively. The `s` for
        "seconds" can be omitted. Only one amount must be specified, though
        combining multiple amounts is optional. The following list contains
        examples of values that are acceptable: `10`, `1d 6h`, `1h 30m`,
        `7d 7d 7d`, `5m 30`.
    api_endpoint: Specifies the handler ID that identifies an endpoint as an API
        endpoint. Calls that terminate here will be handled by the API serving
        framework.

  Special cases:
    When defining a `static_dir` handler, do not use a regular expression in the
    `url` attribute. Both the `url` and `static_dir` attributes are
    automatically mapped to these equivalents::

        <url>/(.*)
        <static_dir>/\\1

    For example, this declaration...::

        url: /images
        static_dir: images_folder

    ...is equivalent to this `static_files` declaration::

        url: /images/(.*)
        static_files: images_folder/\\1
        upload: images_folder/(.*)

  """
    ATTRIBUTES = {HANDLER_STATIC_FILES: validation.Optional(_FILES_REGEX), UPLOAD: validation.Optional(_FILES_REGEX), APPLICATION_READABLE: validation.Optional(bool), HANDLER_STATIC_DIR: validation.Optional(_FILES_REGEX), MIME_TYPE: validation.Optional(str), EXPIRATION: validation.Optional(_EXPIRATION_REGEX), REQUIRE_MATCHING_FILE: validation.Optional(bool), HTTP_HEADERS: validation.Optional(HttpHeadersDict), POSITION: validation.Optional(validation.Options(POSITION_HEAD, POSITION_TAIL)), HANDLER_API_ENDPOINT: validation.Optional(validation.Options((ON, ON_ALIASES), (OFF, OFF_ALIASES))), REDIRECT_HTTP_RESPONSE_CODE: validation.Optional(validation.Options('301', '302', '303', '307'))}
    ATTRIBUTES.update(HandlerBase.ATTRIBUTES)
    COMMON_FIELDS = set([URL, LOGIN, AUTH_FAIL_ACTION, SECURE, REDIRECT_HTTP_RESPONSE_CODE])
    ALLOWED_FIELDS = {HANDLER_STATIC_FILES: (MIME_TYPE, UPLOAD, EXPIRATION, REQUIRE_MATCHING_FILE, HTTP_HEADERS, APPLICATION_READABLE), HANDLER_STATIC_DIR: (MIME_TYPE, EXPIRATION, REQUIRE_MATCHING_FILE, HTTP_HEADERS, APPLICATION_READABLE), HANDLER_SCRIPT: POSITION, HANDLER_API_ENDPOINT: (POSITION, SCRIPT)}

    def GetHandler(self):
        """Gets the handler for a mapping.

    Returns:
      The value of the handler, as determined by the handler ID attribute.
    """
        return getattr(self, self.GetHandlerType())

    def GetHandlerType(self):
        """Gets the handler type of a mapping.

    Returns:
      The handler type as determined by which handler ID attribute is set.

    Raises:
      UnknownHandlerType: If none of the handler ID attributes are set.
      UnexpectedHandlerAttribute: If an unexpected attribute is set for the
          discovered handler type.
      HandlerTypeMissingAttribute: If the handler is missing a required
          attribute for its handler type.
      MissingHandlerAttribute: If a URL handler is missing an attribute.
    """
        if getattr(self, HANDLER_API_ENDPOINT) is not None:
            mapping_type = HANDLER_API_ENDPOINT
        else:
            for id_field in URLMap.ALLOWED_FIELDS:
                if getattr(self, id_field) is not None:
                    mapping_type = id_field
                    break
            else:
                raise appinfo_errors.UnknownHandlerType('Unknown url handler type.\n%s' % str(self))
        allowed_fields = URLMap.ALLOWED_FIELDS[mapping_type]
        for attribute in self.ATTRIBUTES:
            if getattr(self, attribute) is not None and (not (attribute in allowed_fields or attribute in URLMap.COMMON_FIELDS or attribute == mapping_type)):
                raise appinfo_errors.UnexpectedHandlerAttribute('Unexpected attribute "%s" for mapping type %s.' % (attribute, mapping_type))
        if mapping_type == HANDLER_STATIC_FILES and (not self.upload):
            raise appinfo_errors.MissingHandlerAttribute('Missing "%s" attribute for URL "%s".' % (UPLOAD, self.url))
        return mapping_type

    def CheckInitialized(self):
        """Adds additional checking to make sure a handler has correct fields.

    In addition to normal `ValidatedCheck`, this method calls `GetHandlerType`,
    which validates whether all of the handler fields are configured properly.

    Raises:
      UnknownHandlerType: If none of the handler ID attributes are set.
      UnexpectedHandlerAttribute: If an unexpected attribute is set for the
          discovered handler type.
      HandlerTypeMissingAttribute: If the handler is missing a required
          attribute for its handler type.
      ContentTypeSpecifiedMultipleTimes: If `mime_type` is inconsistent with
          `http_headers`.
    """
        super(URLMap, self).CheckInitialized()
        if self.GetHandlerType() in (STATIC_DIR, STATIC_FILES):
            self.AssertUniqueContentType()

    def AssertUniqueContentType(self):
        """Makes sure that `self.http_headers` is consistent with `self.mime_type`.

    This method assumes that `self` is a static handler, either
    `self.static_dir` or `self.static_files`. You cannot specify `None`.

    Raises:
      appinfo_errors.ContentTypeSpecifiedMultipleTimes: If `self.http_headers`
          contains a `Content-Type` header, and `self.mime_type` is set. For
          example, the following configuration would be rejected::

              handlers:
              - url: /static
                static_dir: static
                mime_type: text/html
                http_headers:
                  content-type: text/html


        As this example shows, a configuration will be rejected when
        `http_headers` and `mime_type` specify a content type, even when they
        specify the same content type.
    """
        used_both_fields = self.mime_type and self.http_headers
        if not used_both_fields:
            return
        content_type = self.http_headers.Get('Content-Type')
        if content_type is not None:
            raise appinfo_errors.ContentTypeSpecifiedMultipleTimes('http_header specified a Content-Type header of %r in a handler that also specified a mime_type of %r.' % (content_type, self.mime_type))

    def FixSecureDefaults(self):
        """Forces omitted `secure` handler fields to be set to 'secure: optional'.

    The effect is that `handler.secure` is never equal to the nominal default.
    """
        if self.secure == SECURE_DEFAULT:
            self.secure = SECURE_HTTP_OR_HTTPS

    def WarnReservedURLs(self):
        """Generates a warning for reserved URLs.

    See the `version element documentation`_ to learn which URLs are reserved.

    .. _`version element documentation`:
       https://cloud.google.com/appengine/docs/python/config/appref#syntax
    """
        if self.url == '/form':
            logging.warning('The URL path "/form" is reserved and will not be matched.')

    def ErrorOnPositionForAppInfo(self):
        """Raises an error if position is specified outside of AppInclude objects.

    Raises:
      PositionUsedInAppYamlHandler: If the `position` attribute is specified for
          an `app.yaml` file instead of an `include.yaml` file.
    """
        if self.position:
            raise appinfo_errors.PositionUsedInAppYamlHandler('The position attribute was specified for this handler, but this is an app.yaml file.  Position attribute is only valid for include.yaml files.')