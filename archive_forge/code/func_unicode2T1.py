import reportlab
def unicode2T1(utext, fonts):
    """return a list of (font,string) pairs representing the unicode text"""
    R = []
    font, fonts = (fonts[0], fonts[1:])
    enc = font.encName
    if 'UCS-2' in enc:
        enc = 'UTF16'
    while utext:
        try:
            if isUnicode(utext):
                s = utext.encode(enc)
            else:
                s = utext
            R.append((font, s))
            break
        except UnicodeEncodeError as e:
            i0, il = e.args[2:4]
            if i0:
                R.append((font, utext[:i0].encode(enc)))
            if fonts:
                R.extend(unicode2T1(utext[i0:il], fonts))
            else:
                R.append((font._notdefFont, font._notdefChar * (il - i0)))
            utext = utext[il:]
    return R