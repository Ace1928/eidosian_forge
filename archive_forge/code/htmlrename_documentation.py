from formencode.rewritingparser import RewritingParser

    Add the given prefix to all the fields in the form.

    If dotted is true, then add a dot between prefix and the previous
    name.  Empty fields will use the prefix as the name (with no dot).
    