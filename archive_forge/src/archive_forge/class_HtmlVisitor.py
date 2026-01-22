import sys, re
class HtmlVisitor(SimpleUnicodeVisitor):
    single = dict([(x, 1) for x in 'br,img,area,param,col,hr,meta,link,base,input,frame'.split(',')])
    inline = dict([(x, 1) for x in 'a abbr acronym b basefont bdo big br cite code dfn em font i img input kbd label q s samp select small span strike strong sub sup textarea tt u var'.split(' ')])

    def repr_attribute(self, attrs, name):
        if name == 'class_':
            value = getattr(attrs, name)
            if value is None:
                return
        return super(HtmlVisitor, self).repr_attribute(attrs, name)

    def _issingleton(self, tagname):
        return tagname in self.single

    def _isinline(self, tagname):
        return tagname in self.inline