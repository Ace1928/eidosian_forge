import sys
import os
import re
import docutils
from docutils import frontend, nodes, utils
from docutils.writers import html4css1
from docutils.parsers.rst import directives
class S5HTMLTranslator(html4css1.HTMLTranslator):
    s5_stylesheet_template = '<!-- configuration parameters -->\n<meta name="defaultView" content="%(view_mode)s" />\n<meta name="controlVis" content="%(control_visibility)s" />\n<!-- style sheet links -->\n<script src="%(path)s/slides.js" type="text/javascript"></script>\n<link rel="stylesheet" href="%(path)s/slides.css"\n      type="text/css" media="projection" id="slideProj" />\n<link rel="stylesheet" href="%(path)s/outline.css"\n      type="text/css" media="screen" id="outlineStyle" />\n<link rel="stylesheet" href="%(path)s/print.css"\n      type="text/css" media="print" id="slidePrint" />\n<link rel="stylesheet" href="%(path)s/opera.css"\n      type="text/css" media="projection" id="operaFix" />\n'
    disable_current_slide = '\n<style type="text/css">\n#currentSlide {display: none;}\n</style>\n'
    layout_template = '<div class="layout">\n<div id="controls"></div>\n<div id="currentSlide"></div>\n<div id="header">\n%(header)s\n</div>\n<div id="footer">\n%(title)s%(footer)s\n</div>\n</div>\n'
    default_theme = 'default'
    'Name of the default theme.'
    base_theme_file = '__base__'
    'Name of the file containing the name of the base theme.'
    direct_theme_files = ('slides.css', 'outline.css', 'print.css', 'opera.css', 'slides.js')
    'Names of theme files directly linked to in the output HTML'
    indirect_theme_files = ('s5-core.css', 'framing.css', 'pretty.css', 'blank.gif', 'iepngfix.htc')
    'Names of files used indirectly; imported or used by files in\n    `direct_theme_files`.'
    required_theme_files = indirect_theme_files + direct_theme_files
    'Names of mandatory theme files.'

    def __init__(self, *args):
        html4css1.HTMLTranslator.__init__(self, *args)
        self.theme_file_path = None
        self.setup_theme()
        view_mode = self.document.settings.view_mode
        control_visibility = ('visible', 'hidden')[self.document.settings.hidden_controls]
        self.stylesheet.append(self.s5_stylesheet_template % {'path': self.theme_file_path, 'view_mode': view_mode, 'control_visibility': control_visibility})
        if not self.document.settings.current_slide:
            self.stylesheet.append(self.disable_current_slide)
        self.add_meta('<meta name="version" content="S5 1.1" />\n')
        self.s5_footer = []
        self.s5_header = []
        self.section_count = 0
        self.theme_files_copied = None

    def setup_theme(self):
        if self.document.settings.theme:
            self.copy_theme()
        elif self.document.settings.theme_url:
            self.theme_file_path = self.document.settings.theme_url
        else:
            raise docutils.ApplicationError('No theme specified for S5/HTML writer.')

    def copy_theme(self):
        """
        Locate & copy theme files.

        A theme may be explicitly based on another theme via a '__base__'
        file.  The default base theme is 'default'.  Files are accumulated
        from the specified theme, any base themes, and 'default'.
        """
        settings = self.document.settings
        path = find_theme(settings.theme)
        theme_paths = [path]
        self.theme_files_copied = {}
        required_files_copied = {}
        self.theme_file_path = '%s/%s' % ('ui', settings.theme)
        if settings._destination:
            dest = os.path.join(os.path.dirname(settings._destination), 'ui', settings.theme)
            if not os.path.isdir(dest):
                os.makedirs(dest)
        else:
            return
        default = False
        while path:
            for f in os.listdir(path):
                if f == self.base_theme_file:
                    continue
                if self.copy_file(f, path, dest) and f in self.required_theme_files:
                    required_files_copied[f] = 1
            if default:
                break
            base_theme_file = os.path.join(path, self.base_theme_file)
            if os.path.isfile(base_theme_file):
                lines = open(base_theme_file).readlines()
                for line in lines:
                    line = line.strip()
                    if line and (not line.startswith('#')):
                        path = find_theme(line)
                        if path in theme_paths:
                            path = None
                        else:
                            theme_paths.append(path)
                        break
                else:
                    path = None
            else:
                path = None
            if not path:
                path = find_theme(self.default_theme)
                theme_paths.append(path)
                default = True
        if len(required_files_copied) != len(self.required_theme_files):
            required = list(self.required_theme_files)
            for f in list(required_files_copied.keys()):
                required.remove(f)
            raise docutils.ApplicationError('Theme files not found: %s' % ', '.join(['%r' % f for f in required]))
    files_to_skip_pattern = re.compile('~$|\\.bak$|#$|\\.cvsignore$')

    def copy_file(self, name, source_dir, dest_dir):
        """
        Copy file `name` from `source_dir` to `dest_dir`.
        Return 1 if the file exists in either `source_dir` or `dest_dir`.
        """
        source = os.path.join(source_dir, name)
        dest = os.path.join(dest_dir, name)
        if dest in self.theme_files_copied:
            return 1
        else:
            self.theme_files_copied[dest] = 1
        if os.path.isfile(source):
            if self.files_to_skip_pattern.search(source):
                return None
            settings = self.document.settings
            if os.path.exists(dest) and (not settings.overwrite_theme_files):
                settings.record_dependencies.add(dest)
            else:
                src_file = open(source, 'rb')
                src_data = src_file.read()
                src_file.close()
                dest_file = open(dest, 'wb')
                dest_dir = dest_dir.replace(os.sep, '/')
                dest_file.write(src_data.replace(b'ui/default', dest_dir[dest_dir.rfind('ui/'):].encode(sys.getfilesystemencoding())))
                dest_file.close()
                settings.record_dependencies.add(source)
            return 1
        if os.path.isfile(dest):
            return 1

    def depart_document(self, node):
        self.head_prefix.extend([self.doctype, self.head_prefix_template % {'lang': self.settings.language_code}])
        self.html_prolog.append(self.doctype)
        self.meta.insert(0, self.content_type % self.settings.output_encoding)
        self.head.insert(0, self.content_type % self.settings.output_encoding)
        if self.math_header:
            if self.math_output == 'mathjax':
                self.head.extend(self.math_header)
            else:
                self.stylesheet.extend(self.math_header)
        self.html_head.extend(self.head[1:])
        self.fragment.extend(self.body)
        header = ''.join(self.s5_header)
        footer = ''.join(self.s5_footer)
        title = ''.join(self.html_title).replace('<h1 class="title">', '<h1>')
        layout = self.layout_template % {'header': header, 'title': title, 'footer': footer}
        self.body_prefix.extend(layout)
        self.body_prefix.append('<div class="presentation">\n')
        self.body_prefix.append(self.starttag({'classes': ['slide'], 'ids': ['slide0']}, 'div'))
        if not self.section_count:
            self.body.append('</div>\n')
        self.body_suffix.insert(0, '</div>\n')
        self.html_body.extend(self.body_prefix[1:] + self.body_pre_docinfo + self.docinfo + self.body + self.body_suffix[:-1])

    def depart_footer(self, node):
        start = self.context.pop()
        self.s5_footer.append('<h2>')
        self.s5_footer.extend(self.body[start:])
        self.s5_footer.append('</h2>')
        del self.body[start:]

    def depart_header(self, node):
        start = self.context.pop()
        header = ['<div id="header">\n']
        header.extend(self.body[start:])
        header.append('\n</div>\n')
        del self.body[start:]
        self.s5_header.extend(header)

    def visit_section(self, node):
        if not self.section_count:
            self.body.append('\n</div>\n')
        self.section_count += 1
        self.section_level += 1
        if self.section_level > 1:
            self.body.append(self.starttag(node, 'div', CLASS='section'))
        else:
            self.body.append(self.starttag(node, 'div', CLASS='slide'))

    def visit_subtitle(self, node):
        if isinstance(node.parent, nodes.section):
            level = self.section_level + self.initial_header_level - 1
            if level == 1:
                level = 2
            tag = 'h%s' % level
            self.body.append(self.starttag(node, tag, ''))
            self.context.append('</%s>\n' % tag)
        else:
            html4css1.HTMLTranslator.visit_subtitle(self, node)

    def visit_title(self, node):
        html4css1.HTMLTranslator.visit_title(self, node)