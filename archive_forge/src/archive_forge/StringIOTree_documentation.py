from __future__ import absolute_import  #, unicode_literals

        Returns a new StringIOTree, which is left behind at the current position
        (it what is written to the result will appear right before whatever is
        next written to self).

        Calling getvalue() or copyto() on the result will only return the
        contents written to it.
        