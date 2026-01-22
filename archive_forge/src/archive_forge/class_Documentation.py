from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Documentation(_messages.Message):
    """`Documentation` provides the information for describing a service.
  Example: <pre><code>documentation:   summary: >     The Google Calendar API
  gives access     to most calendar features.   pages:   - name: Overview
  content: &#40;== include google/foo/overview.md ==&#41;   - name: Tutorial
  content: &#40;== include google/foo/tutorial.md ==&#41;     subpages;     -
  name: Java       content: &#40;== include google/foo/tutorial_java.md
  ==&#41;   rules:   - selector: google.calendar.Calendar.Get     description:
  >       ...   - selector: google.calendar.Calendar.Put     description: >
  ... </code></pre> Documentation is provided in markdown syntax. In addition
  to standard markdown features, definition lists, tables and fenced code
  blocks are supported. Section headers can be provided and are interpreted
  relative to the section nesting of the context where a documentation
  fragment is embedded.  Documentation from the IDL is merged with
  documentation defined via the config at normalization time, where
  documentation provided by config rules overrides IDL provided.  A number of
  constructs specific to the API platform are supported in documentation text.
  In order to reference a proto element, the following notation can be used:
  <pre><code>&#91;fully.qualified.proto.name]&#91;]</code></pre> To override
  the display text used for the link, this can be used:
  <pre><code>&#91;display text]&#91;fully.qualified.proto.name]</code></pre>
  Text can be excluded from doc using the following notation:
  <pre><code>&#40;-- internal comment --&#41;</code></pre> Comments can be
  made conditional using a visibility label. The below text will be only
  rendered if the `BETA` label is available: <pre><code>&#40;--BETA: comment
  for BETA users --&#41;</code></pre> A few directives are available in
  documentation. Note that directives must appear on a single line to be
  properly identified. The `include` directive includes a markdown file from
  an external source: <pre><code>&#40;== include path/to/file
  ==&#41;</code></pre> The `resource_for` directive marks a message to be the
  resource of a collection in REST view. If it is not specified, tools attempt
  to infer the resource from the operations in a collection:
  <pre><code>&#40;== resource_for v1.shelves.books ==&#41;</code></pre> The
  directive `suppress_warning` does not directly affect documentation and is
  documented together with service config validation.

  Fields:
    documentationRootUrl: The URL to the root of documentation.
    overview: Declares a single overview page. For example:
      <pre><code>documentation:   summary: ...   overview: &#40;== include
      overview.md ==&#41; </code></pre> This is a shortcut for the following
      declaration (using pages style): <pre><code>documentation:   summary:
      ...   pages:   - name: Overview     content: &#40;== include overview.md
      ==&#41; </code></pre> Note: you cannot specify both `overview` field and
      `pages` field.
    pages: The top level pages for the documentation set.
    rules: Documentation rules for individual elements of the service.
    summary: A short summary of what the service does. Can only be provided by
      plain text.
  """
    documentationRootUrl = _messages.StringField(1)
    overview = _messages.StringField(2)
    pages = _messages.MessageField('Page', 3, repeated=True)
    rules = _messages.MessageField('DocumentationRule', 4, repeated=True)
    summary = _messages.StringField(5)