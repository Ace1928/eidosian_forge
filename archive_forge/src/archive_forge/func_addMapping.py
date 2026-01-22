def addMapping(face, bold, italic, psname):
    """allow a custom font to be put in the mapping"""
    k = (face.lower(), bold, italic)
    _tt2ps_map[k] = psname
    _ps2tt_map[psname.lower()] = k