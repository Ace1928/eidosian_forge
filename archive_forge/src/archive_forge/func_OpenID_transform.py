def OpenID_transform(html, options, _state):
    """
    Replace C{openid.XXX} type C{@rel} attribute values in C{<link>} elements by C{openid:XXX}. The openid URI is also
    added to the top level namespaces with the C{openid:} local name.

    @param html: a DOM node for the top level html element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """
    from ..host import HostLanguage
    if not options.host_language in [HostLanguage.xhtml, HostLanguage.html5, HostLanguage.xhtml5]:
        return
    head = None
    try:
        head = html.getElementsByTagName('head')[0]
    except:
        return
    foundOpenId = False
    for link in html.getElementsByTagName('link'):
        if link.hasAttribute('rel'):
            rel = link.getAttribute('rel')
            newProp = ''
            for n in rel.strip().split():
                if n.startswith('openid.'):
                    newProp += ' ' + n.replace('openid.', 'openid:')
                    foundOpenId = True
                else:
                    newProp += ' ' + n
            link.setAttribute('rel', newProp.strip())
    if foundOpenId and (not head.hasAttribute('xmlns:openid')):
        head.setAttributeNS('', 'xmlns:openid', OPENID_NS)