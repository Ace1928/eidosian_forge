from sympy.utilities.pkgdata import get_resource
from sympy.utilities.decorator import doctest_depends_on
@doctest_depends_on(modules=('lxml',))
def apply_xsl(mml, xsl):
    """Apply a xsl to a MathML string.

    Parameters
    ==========

    mml
        A string with MathML code.
    xsl
        A string representing a path to a xsl (xml stylesheet) file.
        This file name is relative to the PYTHONPATH.

    Examples
    ========

    >>> from sympy.utilities.mathml import apply_xsl
    >>> xsl = 'mathml/data/simple_mmlctop.xsl'
    >>> mml = '<apply> <plus/> <ci>a</ci> <ci>b</ci> </apply>'
    >>> res = apply_xsl(mml,xsl)
    >>> ''.join(res.splitlines())
    '<?xml version="1.0"?><mrow xmlns="http://www.w3.org/1998/Math/MathML">  <mi>a</mi>  <mo> + </mo>  <mi>b</mi></mrow>'
    """
    from lxml import etree
    parser = etree.XMLParser(resolve_entities=False)
    ac = etree.XSLTAccessControl.DENY_ALL
    s = etree.XML(get_resource(xsl).read(), parser=parser)
    transform = etree.XSLT(s, access_control=ac)
    doc = etree.XML(mml, parser=parser)
    result = transform(doc)
    s = str(result)
    return s