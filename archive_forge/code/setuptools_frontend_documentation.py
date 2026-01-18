from __future__ import annotations
from babel.messages import frontend
Catalog merging command for use in ``setup.py`` scripts.

    If correctly installed, this command is available to Setuptools-using
    setup scripts automatically. For projects using plain old ``distutils``,
    the command needs to be registered explicitly in ``setup.py``::

        from babel.messages.setuptools_frontend import update_catalog

        setup(
            ...
            cmdclass = {'update_catalog': update_catalog}
        )

    .. versionadded:: 0.9
    