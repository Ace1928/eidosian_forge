import os
import posixpath
import re
adds autohandler functionality to Mako templates.

requires that the TemplateLookup class is used with templates.

usage::

    <%!
        from mako.ext.autohandler import autohandler
    %>
    <%inherit file="${autohandler(template, context)}"/>


or with custom autohandler filename::

    <%!
        from mako.ext.autohandler import autohandler
    %>
    <%inherit file="${autohandler(template, context, name='somefilename')}"/>

