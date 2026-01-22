import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
class PackageIndex(Environment):
    """A distribution index that scans web pages for download URLs"""

    def __init__(self, index_url='https://pypi.org/simple/', hosts=('*',), ca_bundle=None, verify_ssl=True, *args, **kw):
        super().__init__(*args, **kw)
        self.index_url = index_url + '/'[:not index_url.endswith('/')]
        self.scanned_urls = {}
        self.fetched_urls = {}
        self.package_pages = {}
        self.allows = re.compile('|'.join(map(translate, hosts))).match
        self.to_scan = []
        self.opener = urllib.request.urlopen

    def add(self, dist):
        try:
            parse_version(dist.version)
        except Exception:
            return
        return super().add(dist)

    def process_url(self, url, retrieve=False):
        """Evaluate a URL as a possible download, and maybe retrieve it"""
        if url in self.scanned_urls and (not retrieve):
            return
        self.scanned_urls[url] = True
        if not URL_SCHEME(url):
            self.process_filename(url)
            return
        else:
            dists = list(distros_for_url(url))
            if dists:
                if not self.url_ok(url):
                    return
                self.debug('Found link: %s', url)
        if dists or not retrieve or url in self.fetched_urls:
            list(map(self.add, dists))
            return
        if not self.url_ok(url):
            self.fetched_urls[url] = True
            return
        self.info('Reading %s', url)
        self.fetched_urls[url] = True
        tmpl = 'Download error on %s: %%s -- Some packages may not be found!'
        f = self.open_url(url, tmpl % url)
        if f is None:
            return
        if isinstance(f, urllib.error.HTTPError) and f.code == 401:
            self.info('Authentication error: %s' % f.msg)
        self.fetched_urls[f.url] = True
        if 'html' not in f.headers.get('content-type', '').lower():
            f.close()
            return
        base = f.url
        page = f.read()
        if not isinstance(page, str):
            if isinstance(f, urllib.error.HTTPError):
                charset = 'latin-1'
            else:
                charset = f.headers.get_param('charset') or 'latin-1'
            page = page.decode(charset, 'ignore')
        f.close()
        for match in HREF.finditer(page):
            link = urllib.parse.urljoin(base, htmldecode(match.group(1)))
            self.process_url(link)
        if url.startswith(self.index_url) and getattr(f, 'code', None) != 404:
            page = self.process_index(url, page)

    def process_filename(self, fn, nested=False):
        if not os.path.exists(fn):
            self.warn('Not found: %s', fn)
            return
        if os.path.isdir(fn) and (not nested):
            path = os.path.realpath(fn)
            for item in os.listdir(path):
                self.process_filename(os.path.join(path, item), True)
        dists = distros_for_filename(fn)
        if dists:
            self.debug('Found: %s', fn)
            list(map(self.add, dists))

    def url_ok(self, url, fatal=False):
        s = URL_SCHEME(url)
        is_file = s and s.group(1).lower() == 'file'
        if is_file or self.allows(urllib.parse.urlparse(url)[1]):
            return True
        msg = '\nNote: Bypassing %s (disallowed host; see https://setuptools.pypa.io/en/latest/deprecated/easy_install.html#restricting-downloads-with-allow-hosts for details).\n'
        if fatal:
            raise DistutilsError(msg % url)
        else:
            self.warn(msg, url)

    def scan_egg_links(self, search_path):
        dirs = filter(os.path.isdir, search_path)
        egg_links = ((path, entry) for path in dirs for entry in os.listdir(path) if entry.endswith('.egg-link'))
        list(itertools.starmap(self.scan_egg_link, egg_links))

    def scan_egg_link(self, path, entry):
        with open(os.path.join(path, entry)) as raw_lines:
            lines = list(filter(None, map(str.strip, raw_lines)))
        if len(lines) != 2:
            return
        egg_path, setup_path = lines
        for dist in find_distributions(os.path.join(path, egg_path)):
            dist.location = os.path.join(path, *lines)
            dist.precedence = SOURCE_DIST
            self.add(dist)

    def _scan(self, link):
        NO_MATCH_SENTINEL = (None, None)
        if not link.startswith(self.index_url):
            return NO_MATCH_SENTINEL
        parts = list(map(urllib.parse.unquote, link[len(self.index_url):].split('/')))
        if len(parts) != 2 or '#' in parts[1]:
            return NO_MATCH_SENTINEL
        pkg = safe_name(parts[0])
        ver = safe_version(parts[1])
        self.package_pages.setdefault(pkg.lower(), {})[link] = True
        return (to_filename(pkg), to_filename(ver))

    def process_index(self, url, page):
        """Process the contents of a PyPI page"""
        for match in HREF.finditer(page):
            try:
                self._scan(urllib.parse.urljoin(url, htmldecode(match.group(1))))
            except ValueError:
                pass
        pkg, ver = self._scan(url)
        if not pkg:
            return ''
        for new_url in find_external_links(url, page):
            base, frag = egg_info_for_url(new_url)
            if base.endswith('.py') and (not frag):
                if ver:
                    new_url += '#egg=%s-%s' % (pkg, ver)
                else:
                    self.need_version_info(url)
            self.scan_url(new_url)
        return PYPI_MD5.sub(lambda m: '<a href="%s#md5=%s">%s</a>' % m.group(1, 3, 2), page)

    def need_version_info(self, url):
        self.scan_all('Page at %s links to .py file(s) without version info; an index scan is required.', url)

    def scan_all(self, msg=None, *args):
        if self.index_url not in self.fetched_urls:
            if msg:
                self.warn(msg, *args)
            self.info('Scanning index of all packages (this may take a while)')
        self.scan_url(self.index_url)

    def find_packages(self, requirement):
        self.scan_url(self.index_url + requirement.unsafe_name + '/')
        if not self.package_pages.get(requirement.key):
            self.scan_url(self.index_url + requirement.project_name + '/')
        if not self.package_pages.get(requirement.key):
            self.not_found_in_index(requirement)
        for url in list(self.package_pages.get(requirement.key, ())):
            self.scan_url(url)

    def obtain(self, requirement, installer=None):
        self.prescan()
        self.find_packages(requirement)
        for dist in self[requirement.key]:
            if dist in requirement:
                return dist
            self.debug('%s does not match %s', requirement, dist)
        return super(PackageIndex, self).obtain(requirement, installer)

    def check_hash(self, checker, filename, tfp):
        """
        checker is a ContentChecker
        """
        checker.report(self.debug, 'Validating %%s checksum for %s' % filename)
        if not checker.is_valid():
            tfp.close()
            os.unlink(filename)
            raise DistutilsError('%s validation failed for %s; possible download problem?' % (checker.hash.name, os.path.basename(filename)))

    def add_find_links(self, urls):
        """Add `urls` to the list that will be prescanned for searches"""
        for url in urls:
            if self.to_scan is None or not URL_SCHEME(url) or url.startswith('file:') or list(distros_for_url(url)):
                self.scan_url(url)
            else:
                self.to_scan.append(url)

    def prescan(self):
        """Scan urls scheduled for prescanning (e.g. --find-links)"""
        if self.to_scan:
            list(map(self.scan_url, self.to_scan))
        self.to_scan = None

    def not_found_in_index(self, requirement):
        if self[requirement.key]:
            meth, msg = (self.info, "Couldn't retrieve index page for %r")
        else:
            meth, msg = (self.warn, "Couldn't find index page for %r (maybe misspelled?)")
        meth(msg, requirement.unsafe_name)
        self.scan_all()

    def download(self, spec, tmpdir):
        """Locate and/or download `spec` to `tmpdir`, returning a local path

        `spec` may be a ``Requirement`` object, or a string containing a URL,
        an existing local filename, or a project/version requirement spec
        (i.e. the string form of a ``Requirement`` object).  If it is the URL
        of a .py file with an unambiguous ``#egg=name-version`` tag (i.e., one
        that escapes ``-`` as ``_`` throughout), a trivial ``setup.py`` is
        automatically created alongside the downloaded file.

        If `spec` is a ``Requirement`` object or a string containing a
        project/version requirement spec, this method returns the location of
        a matching distribution (possibly after downloading it to `tmpdir`).
        If `spec` is a locally existing file or directory name, it is simply
        returned unchanged.  If `spec` is a URL, it is downloaded to a subpath
        of `tmpdir`, and the local filename is returned.  Various errors may be
        raised if a problem occurs during downloading.
        """
        if not isinstance(spec, Requirement):
            scheme = URL_SCHEME(spec)
            if scheme:
                found = self._download_url(scheme.group(1), spec, tmpdir)
                base, fragment = egg_info_for_url(spec)
                if base.endswith('.py'):
                    found = self.gen_setup(found, fragment, tmpdir)
                return found
            elif os.path.exists(spec):
                return spec
            else:
                spec = parse_requirement_arg(spec)
        return getattr(self.fetch_distribution(spec, tmpdir), 'location', None)

    def fetch_distribution(self, requirement, tmpdir, force_scan=False, source=False, develop_ok=False, local_index=None):
        """Obtain a distribution suitable for fulfilling `requirement`

        `requirement` must be a ``pkg_resources.Requirement`` instance.
        If necessary, or if the `force_scan` flag is set, the requirement is
        searched for in the (online) package index as well as the locally
        installed packages.  If a distribution matching `requirement` is found,
        the returned distribution's ``location`` is the value you would have
        gotten from calling the ``download()`` method with the matching
        distribution's URL or filename.  If no matching distribution is found,
        ``None`` is returned.

        If the `source` flag is set, only source distributions and source
        checkout links will be considered.  Unless the `develop_ok` flag is
        set, development and system eggs (i.e., those using the ``.egg-info``
        format) will be ignored.
        """
        self.info('Searching for %s', requirement)
        skipped = {}
        dist = None

        def find(req, env=None):
            if env is None:
                env = self
            for dist in env[req.key]:
                if dist.precedence == DEVELOP_DIST and (not develop_ok):
                    if dist not in skipped:
                        self.warn('Skipping development or system egg: %s', dist)
                        skipped[dist] = 1
                    continue
                test = dist in req and (dist.precedence <= SOURCE_DIST or not source)
                if test:
                    loc = self.download(dist.location, tmpdir)
                    dist.download_location = loc
                    if os.path.exists(dist.download_location):
                        return dist
        if force_scan:
            self.prescan()
            self.find_packages(requirement)
            dist = find(requirement)
        if not dist and local_index is not None:
            dist = find(requirement, local_index)
        if dist is None:
            if self.to_scan is not None:
                self.prescan()
            dist = find(requirement)
        if dist is None and (not force_scan):
            self.find_packages(requirement)
            dist = find(requirement)
        if dist is None:
            self.warn('No local packages or working download links found for %s%s', source and 'a source distribution of ' or '', requirement)
        else:
            self.info('Best match: %s', dist)
            return dist.clone(location=dist.download_location)

    def fetch(self, requirement, tmpdir, force_scan=False, source=False):
        """Obtain a file suitable for fulfilling `requirement`

        DEPRECATED; use the ``fetch_distribution()`` method now instead.  For
        backward compatibility, this routine is identical but returns the
        ``location`` of the downloaded distribution instead of a distribution
        object.
        """
        dist = self.fetch_distribution(requirement, tmpdir, force_scan, source)
        if dist is not None:
            return dist.location
        return None

    def gen_setup(self, filename, fragment, tmpdir):
        match = EGG_FRAGMENT.match(fragment)
        dists = match and [d for d in interpret_distro_name(filename, match.group(1), None) if d.version] or []
        if len(dists) == 1:
            basename = os.path.basename(filename)
            if os.path.dirname(filename) != tmpdir:
                dst = os.path.join(tmpdir, basename)
                if not (os.path.exists(dst) and os.path.samefile(filename, dst)):
                    shutil.copy2(filename, dst)
                    filename = dst
            with open(os.path.join(tmpdir, 'setup.py'), 'w') as file:
                file.write('from setuptools import setup\nsetup(name=%r, version=%r, py_modules=[%r])\n' % (dists[0].project_name, dists[0].version, os.path.splitext(basename)[0]))
            return filename
        elif match:
            raise DistutilsError("Can't unambiguously interpret project/version identifier %r; any dashes in the name or version should be escaped using underscores. %r" % (fragment, dists))
        else:
            raise DistutilsError("Can't process plain .py files without an '#egg=name-version' suffix to enable automatic setup script generation.")
    dl_blocksize = 8192

    def _download_to(self, url, filename):
        self.info('Downloading %s', url)
        fp = None
        try:
            checker = HashChecker.from_url(url)
            fp = self.open_url(url)
            if isinstance(fp, urllib.error.HTTPError):
                raise DistutilsError("Can't download %s: %s %s" % (url, fp.code, fp.msg))
            headers = fp.info()
            blocknum = 0
            bs = self.dl_blocksize
            size = -1
            if 'content-length' in headers:
                sizes = headers.get_all('Content-Length')
                size = max(map(int, sizes))
                self.reporthook(url, filename, blocknum, bs, size)
            with open(filename, 'wb') as tfp:
                while True:
                    block = fp.read(bs)
                    if block:
                        checker.feed(block)
                        tfp.write(block)
                        blocknum += 1
                        self.reporthook(url, filename, blocknum, bs, size)
                    else:
                        break
                self.check_hash(checker, filename, tfp)
            return headers
        finally:
            if fp:
                fp.close()

    def reporthook(self, url, filename, blocknum, blksize, size):
        pass

    def open_url(self, url, warning=None):
        if url.startswith('file:'):
            return local_open(url)
        try:
            return open_with_auth(url, self.opener)
        except (ValueError, http.client.InvalidURL) as v:
            msg = ' '.join([str(arg) for arg in v.args])
            if warning:
                self.warn(warning, msg)
            else:
                raise DistutilsError('%s %s' % (url, msg)) from v
        except urllib.error.HTTPError as v:
            return v
        except urllib.error.URLError as v:
            if warning:
                self.warn(warning, v.reason)
            else:
                raise DistutilsError('Download error for %s: %s' % (url, v.reason)) from v
        except http.client.BadStatusLine as v:
            if warning:
                self.warn(warning, v.line)
            else:
                raise DistutilsError('%s returned a bad status line. The server might be down, %s' % (url, v.line)) from v
        except (http.client.HTTPException, socket.error) as v:
            if warning:
                self.warn(warning, v)
            else:
                raise DistutilsError('Download error for %s: %s' % (url, v)) from v

    def _download_url(self, scheme, url, tmpdir):
        name, fragment = egg_info_for_url(url)
        if name:
            while '..' in name:
                name = name.replace('..', '.').replace('\\', '_')
        else:
            name = '__downloaded__'
        if name.endswith('.egg.zip'):
            name = name[:-4]
        filename = os.path.join(tmpdir, name)
        if scheme == 'svn' or scheme.startswith('svn+'):
            return self._download_svn(url, filename)
        elif scheme == 'git' or scheme.startswith('git+'):
            return self._download_git(url, filename)
        elif scheme.startswith('hg+'):
            return self._download_hg(url, filename)
        elif scheme == 'file':
            return urllib.request.url2pathname(urllib.parse.urlparse(url)[2])
        else:
            self.url_ok(url, True)
            return self._attempt_download(url, filename)

    def scan_url(self, url):
        self.process_url(url, True)

    def _attempt_download(self, url, filename):
        headers = self._download_to(url, filename)
        if 'html' in headers.get('content-type', '').lower():
            return self._invalid_download_html(url, headers, filename)
        else:
            return filename

    def _invalid_download_html(self, url, headers, filename):
        os.unlink(filename)
        raise DistutilsError(f'Unexpected HTML page found at {url}')

    def _download_svn(self, url, _filename):
        raise DistutilsError(f'Invalid config, SVN download is not supported: {url}')

    @staticmethod
    def _vcs_split_rev_from_url(url, pop_prefix=False):
        scheme, netloc, path, query, frag = urllib.parse.urlsplit(url)
        scheme = scheme.split('+', 1)[-1]
        path = path.split('#', 1)[0]
        rev = None
        if '@' in path:
            path, rev = path.rsplit('@', 1)
        url = urllib.parse.urlunsplit((scheme, netloc, path, query, ''))
        return (url, rev)

    def _download_git(self, url, filename):
        filename = filename.split('#', 1)[0]
        url, rev = self._vcs_split_rev_from_url(url, pop_prefix=True)
        self.info('Doing git clone from %s to %s', url, filename)
        os.system('git clone --quiet %s %s' % (url, filename))
        if rev is not None:
            self.info('Checking out %s', rev)
            os.system('git -C %s checkout --quiet %s' % (filename, rev))
        return filename

    def _download_hg(self, url, filename):
        filename = filename.split('#', 1)[0]
        url, rev = self._vcs_split_rev_from_url(url, pop_prefix=True)
        self.info('Doing hg clone from %s to %s', url, filename)
        os.system('hg clone --quiet %s %s' % (url, filename))
        if rev is not None:
            self.info('Updating to %s', rev)
            os.system('hg --cwd %s up -C -r %s -q' % (filename, rev))
        return filename

    def debug(self, msg, *args):
        log.debug(msg, *args)

    def info(self, msg, *args):
        log.info(msg, *args)

    def warn(self, msg, *args):
        log.warn(msg, *args)