import optparse

A subclass of ``optparse.OptionParser`` that allows boolean long
options (like ``--verbose``) to also take arguments (like
``--verbose=true``).  Arguments *must* use ``=``.
