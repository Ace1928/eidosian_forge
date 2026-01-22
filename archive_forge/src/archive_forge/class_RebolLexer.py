import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RebolLexer(RegexLexer):
    """
    A `REBOL <http://www.rebol.com/>`_ lexer.

    .. versionadded:: 1.1
    """
    name = 'REBOL'
    aliases = ['rebol']
    filenames = ['*.r', '*.r3', '*.reb']
    mimetypes = ['text/x-rebol']
    flags = re.IGNORECASE | re.MULTILINE
    escape_re = '(?:\\^\\([0-9a-f]{1,4}\\)*)'

    def word_callback(lexer, match):
        word = match.group()
        if re.match('.*:$', word):
            yield (match.start(), Generic.Subheading, word)
        elif re.match('(native|alias|all|any|as-string|as-binary|bind|bound\\?|case|catch|checksum|comment|debase|dehex|exclude|difference|disarm|either|else|enbase|foreach|remove-each|form|free|get|get-env|if|in|intersect|loop|minimum-of|maximum-of|mold|new-line|new-line\\?|not|now|prin|print|reduce|compose|construct|repeat|reverse|save|script\\?|set|shift|switch|throw|to-hex|trace|try|type\\?|union|unique|unless|unprotect|unset|until|use|value\\?|while|compress|decompress|secure|open|close|read|read-io|write-io|write|update|query|wait|input\\?|exp|log-10|log-2|log-e|square-root|cosine|sine|tangent|arccosine|arcsine|arctangent|protect|lowercase|uppercase|entab|detab|connected\\?|browse|launch|stats|get-modes|set-modes|to-local-file|to-rebol-file|encloak|decloak|create-link|do-browser|bind\\?|hide|draw|show|size-text|textinfo|offset-to-caret|caret-to-offset|local-request-file|rgb-to-hsv|hsv-to-rgb|crypt-strength\\?|dh-make-key|dh-generate-key|dh-compute-key|dsa-make-key|dsa-generate-key|dsa-make-signature|dsa-verify-signature|rsa-make-key|rsa-generate-key|rsa-encrypt)$', word):
            yield (match.start(), Name.Builtin, word)
        elif re.match('(add|subtract|multiply|divide|remainder|power|and~|or~|xor~|minimum|maximum|negate|complement|absolute|random|head|tail|next|back|skip|at|pick|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|path|find|select|make|to|copy\\*|insert|remove|change|poke|clear|trim|sort|min|max|abs|cp|copy)$', word):
            yield (match.start(), Name.Function, word)
        elif re.match('(error|source|input|license|help|install|echo|Usage|with|func|throw-on-error|function|does|has|context|probe|\\?\\?|as-pair|mod|modulo|round|repend|about|set-net|append|join|rejoin|reform|remold|charset|array|replace|move|extract|forskip|forall|alter|first+|also|take|for|forever|dispatch|attempt|what-dir|change-dir|clean-path|list-dir|dirize|rename|split-path|delete|make-dir|delete-dir|in-dir|confirm|dump-obj|upgrade|what|build-tag|process-source|build-markup|decode-cgi|read-cgi|write-user|save-user|set-user-name|protect-system|parse-xml|cvs-date|cvs-version|do-boot|get-net-info|desktop|layout|scroll-para|get-face|alert|set-face|uninstall|unfocus|request-dir|center-face|do-events|net-error|decode-url|parse-header|parse-header-date|parse-email-addrs|import-email|send|build-attach-body|resend|show-popup|hide-popup|open-events|find-key-face|do-face|viewtop|confine|find-window|insert-event-func|remove-event-func|inform|dump-pane|dump-face|flag-face|deflag-face|clear-fields|read-net|vbug|path-thru|read-thru|load-thru|do-thru|launch-thru|load-image|request-download|do-face-alt|set-font|set-para|get-style|set-style|make-face|stylize|choose|hilight-text|hilight-all|unlight-text|focus|scroll-drag|clear-face|reset-face|scroll-face|resize-face|load-stock|load-stock-block|notify|request|flash|request-color|request-pass|request-text|request-list|request-date|request-file|dbug|editor|link-relative-path|emailer|parse-error)$', word):
            yield (match.start(), Keyword.Namespace, word)
        elif re.match('(halt|quit|do|load|q|recycle|call|run|ask|parse|view|unview|return|exit|break)$', word):
            yield (match.start(), Name.Exception, word)
        elif re.match('REBOL$', word):
            yield (match.start(), Generic.Heading, word)
        elif re.match('to-.*', word):
            yield (match.start(), Keyword, word)
        elif re.match('(\\+|-|\\*|/|//|\\*\\*|and|or|xor|=\\?|=|==|<>|<|>|<=|>=)$', word):
            yield (match.start(), Operator, word)
        elif re.match('.*\\?$', word):
            yield (match.start(), Keyword, word)
        elif re.match('.*\\!$', word):
            yield (match.start(), Keyword.Type, word)
        elif re.match("'.*", word):
            yield (match.start(), Name.Variable.Instance, word)
        elif re.match('#.*', word):
            yield (match.start(), Name.Label, word)
        elif re.match('%.*', word):
            yield (match.start(), Name.Decorator, word)
        else:
            yield (match.start(), Name.Variable, word)
    tokens = {'root': [('[^R]+', Comment), ('REBOL\\s+\\[', Generic.Strong, 'script'), ('R', Comment)], 'script': [('\\s+', Text), ('#"', String.Char, 'char'), ('#\\{[0-9a-f]*\\}', Number.Hex), ('2#\\{', Number.Hex, 'bin2'), ('64#\\{[0-9a-z+/=\\s]*\\}', Number.Hex), ('"', String, 'string'), ('\\{', String, 'string2'), (';#+.*\\n', Comment.Special), (';\\*+.*\\n', Comment.Preproc), (';.*\\n', Comment), ('%"', Name.Decorator, 'stringFile'), ('%[^(^{")\\s\\[\\]]+', Name.Decorator), ('[+-]?([a-z]{1,3})?\\$\\d+(\\.\\d+)?', Number.Float), ('[+-]?\\d+\\:\\d+(\\:\\d+)?(\\.\\d+)?', String.Other), ('\\d+[\\-/][0-9a-z]+[\\-/]\\d+(\\/\\d+\\:\\d+((\\:\\d+)?([.\\d+]?([+-]?\\d+:\\d+)?)?)?)?', String.Other), ('\\d+(\\.\\d+)+\\.\\d+', Keyword.Constant), ('\\d+X\\d+', Keyword.Constant), ("[+-]?\\d+(\\'\\d+)?([.,]\\d*)?E[+-]?\\d+", Number.Float), ("[+-]?\\d+(\\'\\d+)?[.,]\\d*", Number.Float), ("[+-]?\\d+(\\'\\d+)?", Number), ('[\\[\\]()]', Generic.Strong), ('[a-z]+[^(^{"\\s:)]*://[^(^{"\\s)]*', Name.Decorator), ('mailto:[^(^{"@\\s)]+@[^(^{"@\\s)]+', Name.Decorator), ('[^(^{"@\\s)]+@[^(^{"@\\s)]+', Name.Decorator), ('comment\\s"', Comment, 'commentString1'), ('comment\\s\\{', Comment, 'commentString2'), ('comment\\s\\[', Comment, 'commentBlock'), ('comment\\s[^(\\s{"\\[]+', Comment), ('/[^(^{")\\s/[\\]]*', Name.Attribute), ('([^(^{")\\s/[\\]]+)(?=[:({"\\s/\\[\\]])', word_callback), ('<[\\w:.-]*>', Name.Tag), ('<[^(<>\\s")]+', Name.Tag, 'tag'), ('([^(^{")\\s]+)', Text)], 'string': [('[^(^")]+', String), (escape_re, String.Escape), ('[(|)]+', String), ('\\^.', String.Escape), ('"', String, '#pop')], 'string2': [('[^(^{})]+', String), (escape_re, String.Escape), ('[(|)]+', String), ('\\^.', String.Escape), ('\\{', String, '#push'), ('\\}', String, '#pop')], 'stringFile': [('[^(^")]+', Name.Decorator), (escape_re, Name.Decorator), ('\\^.', Name.Decorator), ('"', Name.Decorator, '#pop')], 'char': [(escape_re + '"', String.Char, '#pop'), ('\\^."', String.Char, '#pop'), ('."', String.Char, '#pop')], 'tag': [(escape_re, Name.Tag), ('"', Name.Tag, 'tagString'), ('[^(<>\\r\\n")]+', Name.Tag), ('>', Name.Tag, '#pop')], 'tagString': [('[^(^")]+', Name.Tag), (escape_re, Name.Tag), ('[(|)]+', Name.Tag), ('\\^.', Name.Tag), ('"', Name.Tag, '#pop')], 'tuple': [('(\\d+\\.)+', Keyword.Constant), ('\\d+', Keyword.Constant, '#pop')], 'bin2': [('\\s+', Number.Hex), ('([01]\\s*){8}', Number.Hex), ('\\}', Number.Hex, '#pop')], 'commentString1': [('[^(^")]+', Comment), (escape_re, Comment), ('[(|)]+', Comment), ('\\^.', Comment), ('"', Comment, '#pop')], 'commentString2': [('[^(^{})]+', Comment), (escape_re, Comment), ('[(|)]+', Comment), ('\\^.', Comment), ('\\{', Comment, '#push'), ('\\}', Comment, '#pop')], 'commentBlock': [('\\[', Comment, '#push'), ('\\]', Comment, '#pop'), ('"', Comment, 'commentString1'), ('\\{', Comment, 'commentString2'), ('[^(\\[\\]"{)]+', Comment)]}

    def analyse_text(text):
        """
        Check if code contains REBOL header and so it probably not R code
        """
        if re.match('^\\s*REBOL\\s*\\[', text, re.IGNORECASE):
            return 1.0
        elif re.search('\\s*REBOL\\s*[', text, re.IGNORECASE):
            return 0.5