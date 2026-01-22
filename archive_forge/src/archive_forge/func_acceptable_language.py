def acceptable_language(accept_header, server_languages, ignore_wildcard=True, assume_superiors=True):
    """Determines if the given language is acceptable to the user agent.

    The accept_header should be the value present in the HTTP
    "Accept-Language:" header.  In mod_python this is typically
    obtained from the req.http_headers_in table; in WSGI it is
    environ["Accept-Language"]; other web frameworks may provide other
    methods of obtaining it.

    Optionally the accept_header parameter can be pre-parsed, as
    returned by the parse_accept_language_header() function defined in
    this module.

    The server_languages argument should either be a single language
    string, a language_tag object, or a sequence of them.  It
    represents the set of languages that the server is willing to
    send to the user agent.

    Note that the wildcarded language tag "*" will be ignored.  To
    override this, call with ignore_wildcard=False, and even then
    it will be the lowest-priority choice regardless of it's
    quality factor (as per HTTP spec).

    If the assume_superiors is True then it the languages that the
    browser accepts will automatically include all superior languages.
    Any superior languages which must be added are done so with one
    half the qvalue of the language which is present.  For example, if
    the accept string is "en-US", then it will be treated as if it
    were "en-US, en;q=0.5".  Note that although the HTTP 1.1 spec says
    that browsers are supposed to encourage users to configure all
    acceptable languages, sometimes they don't, thus the ability
    for this function to assume this.  But setting assume_superiors
    to False will insure strict adherence to the HTTP 1.1 spec; which
    means that if the browser accepts "en-US", then it will not
    be acceptable to send just "en" to it.

    This function returns the language which is the most prefered and
    is acceptable to both the user agent and the caller.  It will
    return None if no language is negotiable, otherwise the return
    value is always an instance of language_tag.

    See also: RFC 3066 <http://www.ietf.org/rfc/rfc3066.txt>, and
    ISO 639, links at <http://en.wikipedia.org/wiki/ISO_639>, and
    <http://www.iana.org/assignments/language-tags>.
    
    """
    if _is_string(accept_header):
        accept_list = parse_accept_language_header(accept_header)
    else:
        accept_list = accept_header
    accept_list.sort()
    all_tags = [a[0] for a in accept_list]
    if assume_superiors:
        to_add = []
        for langtag, qvalue, _args in accept_list:
            if len(langtag) >= 2:
                for suptag in langtag.all_superiors(include_wildcard=False):
                    if suptag not in all_tags:
                        to_add.append((suptag, qvalue / 2, ''))
                        all_tags.append(suptag)
        accept_list.extend(to_add)
    if _is_string(server_languages):
        server_languages = [language_tag(server_languages)]
    elif isinstance(server_languages, language_tag):
        server_languages = [server_languages]
    else:
        server_languages = [language_tag(lang) for lang in server_languages]
    best = None
    for langtag, qvalue, _args in accept_list:
        if qvalue <= 0:
            continue
        if ignore_wildcard and langtag.is_universal_wildcard():
            continue
        for svrlang in server_languages:
            matchlen = -1
            if svrlang.dialect_of(langtag, ignore_wildcard=ignore_wildcard):
                matchlen = len(langtag)
                if not best or matchlen > best[2] or (matchlen == best[2] and qvalue > best[1]):
                    best = (langtag, qvalue, matchlen)
    if not best:
        return None
    return best[0]