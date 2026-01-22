import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
class Remote(LazyMixin, IterableObj):
    """Provides easy read and write access to a git remote.

    Everything not part of this interface is considered an option for the current
    remote, allowing constructs like remote.pushurl to query the pushurl.

    :note: When querying configuration, the configuration accessor will be cached
        to speed up subsequent accesses.
    """
    __slots__ = ('repo', 'name', '_config_reader')
    _id_attribute_ = 'name'
    unsafe_git_fetch_options = ['--upload-pack']
    unsafe_git_pull_options = ['--upload-pack']
    unsafe_git_push_options = ['--receive-pack', '--exec']

    def __init__(self, repo: 'Repo', name: str) -> None:
        """Initialize a remote instance.

        :param repo: The repository we are a remote of
        :param name: The name of the remote, e.g. 'origin'
        """
        self.repo = repo
        self.name = name
        self.url: str

    def __getattr__(self, attr: str) -> Any:
        """Allows to call this instance like
        remote.special( \\*args, \\*\\*kwargs) to call git-remote special self.name."""
        if attr == '_config_reader':
            return super().__getattr__(attr)
        try:
            return self._config_reader.get(attr)
        except cp.NoOptionError:
            return super().__getattr__(attr)

    def _config_section_name(self) -> str:
        return 'remote "%s"' % self.name

    def _set_cache_(self, attr: str) -> None:
        if attr == '_config_reader':
            self._config_reader = SectionConstraint(self.repo.config_reader('repository'), self._config_section_name())
        else:
            super()._set_cache_(attr)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return '<git.%s "%s">' % (self.__class__.__name__, self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.name)

    def exists(self) -> bool:
        """
        :return: True if this is a valid, existing remote.
            Valid remotes have an entry in the repository's configuration.
        """
        try:
            self.config_reader.get('url')
            return True
        except cp.NoOptionError:
            return True
        except cp.NoSectionError:
            return False

    @classmethod
    def iter_items(cls, repo: 'Repo', *args: Any, **kwargs: Any) -> Iterator['Remote']:
        """:return: Iterator yielding Remote objects of the given repository"""
        for section in repo.config_reader('repository').sections():
            if not section.startswith('remote '):
                continue
            lbound = section.find('"')
            rbound = section.rfind('"')
            if lbound == -1 or rbound == -1:
                raise ValueError('Remote-Section has invalid format: %r' % section)
            yield Remote(repo, section[lbound + 1:rbound])

    def set_url(self, new_url: str, old_url: Optional[str]=None, allow_unsafe_protocols: bool=False, **kwargs: Any) -> 'Remote':
        """Configure URLs on current remote (cf command git remote set_url).

        This command manages URLs on the remote.

        :param new_url: String being the URL to add as an extra remote URL
        :param old_url: When set, replaces this URL with new_url for the remote
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :return: self
        """
        if not allow_unsafe_protocols:
            Git.check_unsafe_protocols(new_url)
        scmd = 'set-url'
        kwargs['insert_kwargs_after'] = scmd
        if old_url:
            self.repo.git.remote(scmd, '--', self.name, new_url, old_url, **kwargs)
        else:
            self.repo.git.remote(scmd, '--', self.name, new_url, **kwargs)
        return self

    def add_url(self, url: str, allow_unsafe_protocols: bool=False, **kwargs: Any) -> 'Remote':
        """Adds a new url on current remote (special case of git remote set_url).

        This command adds new URLs to a given remote, making it possible to have
        multiple URLs for a single remote.

        :param url: String being the URL to add as an extra remote URL
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :return: self
        """
        return self.set_url(url, add=True, allow_unsafe_protocols=allow_unsafe_protocols)

    def delete_url(self, url: str, **kwargs: Any) -> 'Remote':
        """Deletes a new url on current remote (special case of git remote set_url)

        This command deletes new URLs to a given remote, making it possible to have
        multiple URLs for a single remote.

        :param url: String being the URL to delete from the remote
        :return: self
        """
        return self.set_url(url, delete=True)

    @property
    def urls(self) -> Iterator[str]:
        """:return: Iterator yielding all configured URL targets on a remote as strings"""
        try:
            remote_details = self.repo.git.remote('get-url', '--all', self.name)
            assert isinstance(remote_details, str)
            for line in remote_details.split('\n'):
                yield line
        except GitCommandError as ex:
            if 'Unknown subcommand: get-url' in str(ex):
                try:
                    remote_details = self.repo.git.remote('show', self.name)
                    assert isinstance(remote_details, str)
                    for line in remote_details.split('\n'):
                        if '  Push  URL:' in line:
                            yield line.split(': ')[-1]
                except GitCommandError as _ex:
                    if any((msg in str(_ex) for msg in ['correct access rights', 'cannot run ssh'])):
                        remote_details = self.repo.git.config('--get-all', 'remote.%s.url' % self.name)
                        assert isinstance(remote_details, str)
                        for line in remote_details.split('\n'):
                            yield line
                    else:
                        raise _ex
            else:
                raise ex

    @property
    def refs(self) -> IterableList[RemoteReference]:
        """
        :return:
            IterableList of RemoteReference objects. It is prefixed, allowing
            you to omit the remote path portion, e.g.::
            remote.refs.master # yields RemoteReference('/refs/remotes/origin/master')
        """
        out_refs: IterableList[RemoteReference] = IterableList(RemoteReference._id_attribute_, '%s/' % self.name)
        out_refs.extend(RemoteReference.list_items(self.repo, remote=self.name))
        return out_refs

    @property
    def stale_refs(self) -> IterableList[Reference]:
        """
        :return:
            IterableList RemoteReference objects that do not have a corresponding
            head in the remote reference anymore as they have been deleted on the
            remote side, but are still available locally.

            The IterableList is prefixed, hence the 'origin' must be omitted. See
            'refs' property for an example.

            To make things more complicated, it can be possible for the list to include
            other kinds of references, for example, tag references, if these are stale
            as well. This is a fix for the issue described here:
            https://github.com/gitpython-developers/GitPython/issues/260
        """
        out_refs: IterableList[Reference] = IterableList(RemoteReference._id_attribute_, '%s/' % self.name)
        for line in self.repo.git.remote('prune', '--dry-run', self).splitlines()[2:]:
            token = ' * [would prune] '
            if not line.startswith(token):
                continue
            ref_name = line.replace(token, '')
            if ref_name.startswith(Reference._common_path_default + '/'):
                out_refs.append(Reference.from_path(self.repo, ref_name))
            else:
                fqhn = '%s/%s' % (RemoteReference._common_path_default, ref_name)
                out_refs.append(RemoteReference(self.repo, fqhn))
        return out_refs

    @classmethod
    def create(cls, repo: 'Repo', name: str, url: str, allow_unsafe_protocols: bool=False, **kwargs: Any) -> 'Remote':
        """Create a new remote to the given repository.

        :param repo: Repository instance that is to receive the new remote
        :param name: Desired name of the remote
        :param url: URL which corresponds to the remote's name
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :param kwargs: Additional arguments to be passed to the git-remote add command
        :return: New Remote instance
        :raise GitCommandError: in case an origin with that name already exists
        """
        scmd = 'add'
        kwargs['insert_kwargs_after'] = scmd
        url = Git.polish_url(url)
        if not allow_unsafe_protocols:
            Git.check_unsafe_protocols(url)
        repo.git.remote(scmd, '--', name, url, **kwargs)
        return cls(repo, name)

    @classmethod
    def add(cls, repo: 'Repo', name: str, url: str, **kwargs: Any) -> 'Remote':
        return cls.create(repo, name, url, **kwargs)

    @classmethod
    def remove(cls, repo: 'Repo', name: str) -> str:
        """Remove the remote with the given name.

        :return: The passed remote name to remove
        """
        repo.git.remote('rm', name)
        if isinstance(name, cls):
            name._clear_cache()
        return name
    rm = remove

    def rename(self, new_name: str) -> 'Remote':
        """Rename self to the given new_name.

        :return: self
        """
        if self.name == new_name:
            return self
        self.repo.git.remote('rename', self.name, new_name)
        self.name = new_name
        self._clear_cache()
        return self

    def update(self, **kwargs: Any) -> 'Remote':
        """Fetch all changes for this remote, including new branches which will
        be forced in (in case your local remote branch is not part the new remote
        branch's ancestry anymore).

        :param kwargs: Additional arguments passed to git-remote update
        :return: self
        """
        scmd = 'update'
        kwargs['insert_kwargs_after'] = scmd
        self.repo.git.remote(scmd, self.name, **kwargs)
        return self

    def _get_fetch_info_from_stderr(self, proc: 'Git.AutoInterrupt', progress: Union[Callable[..., Any], RemoteProgress, None], kill_after_timeout: Union[None, float]=None) -> IterableList['FetchInfo']:
        progress = to_progress_instance(progress)
        output: IterableList['FetchInfo'] = IterableList('name')
        fetch_info_lines = []
        cmds = set(FetchInfo._flag_map.keys())
        progress_handler = progress.new_message_handler()
        handle_process_output(proc, None, progress_handler, finalizer=None, decode_streams=False, kill_after_timeout=kill_after_timeout)
        stderr_text = progress.error_lines and '\n'.join(progress.error_lines) or ''
        proc.wait(stderr=stderr_text)
        if stderr_text:
            _logger.warning('Error lines received while fetching: %s', stderr_text)
        for line in progress.other_lines:
            line = force_text(line)
            for cmd in cmds:
                if len(line) > 1 and line[0] == ' ' and (line[1] == cmd):
                    fetch_info_lines.append(line)
                    continue
        fetch_head = SymbolicReference(self.repo, 'FETCH_HEAD')
        with open(fetch_head.abspath, 'rb') as fp:
            fetch_head_info = [line.decode(defenc) for line in fp.readlines()]
        l_fil = len(fetch_info_lines)
        l_fhi = len(fetch_head_info)
        if l_fil != l_fhi:
            msg = 'Fetch head lines do not match lines provided via progress information\n'
            msg += 'length of progress lines %i should be equal to lines in FETCH_HEAD file %i\n'
            msg += 'Will ignore extra progress lines or fetch head lines.'
            msg %= (l_fil, l_fhi)
            _logger.debug(msg)
            _logger.debug(b'info lines: ' + str(fetch_info_lines).encode('UTF-8'))
            _logger.debug(b'head info: ' + str(fetch_head_info).encode('UTF-8'))
            if l_fil < l_fhi:
                fetch_head_info = fetch_head_info[:l_fil]
            else:
                fetch_info_lines = fetch_info_lines[:l_fhi]
        for err_line, fetch_line in zip(fetch_info_lines, fetch_head_info):
            try:
                output.append(FetchInfo._from_line(self.repo, err_line, fetch_line))
            except ValueError as exc:
                _logger.debug('Caught error while parsing line: %s', exc)
                _logger.warning('Git informed while fetching: %s', err_line.strip())
        return output

    def _get_push_info(self, proc: 'Git.AutoInterrupt', progress: Union[Callable[..., Any], RemoteProgress, None], kill_after_timeout: Union[None, float]=None) -> PushInfoList:
        progress = to_progress_instance(progress)
        progress_handler = progress.new_message_handler()
        output: PushInfoList = PushInfoList()

        def stdout_handler(line: str) -> None:
            try:
                output.append(PushInfo._from_line(self, line))
            except ValueError:
                pass
        handle_process_output(proc, stdout_handler, progress_handler, finalizer=None, decode_streams=False, kill_after_timeout=kill_after_timeout)
        stderr_text = progress.error_lines and '\n'.join(progress.error_lines) or ''
        try:
            proc.wait(stderr=stderr_text)
        except Exception as e:
            if not output:
                raise
            elif stderr_text:
                _logger.warning('Error lines received while fetching: %s', stderr_text)
                output.error = e
        return output

    def _assert_refspec(self) -> None:
        """Turns out we can't deal with remotes if the refspec is missing."""
        config = self.config_reader
        unset = 'placeholder'
        try:
            if config.get_value('fetch', default=unset) is unset:
                msg = "Remote '%s' has no refspec set.\n"
                msg += 'You can set it as follows:'
                msg += ' \'git config --add "remote.%s.fetch +refs/heads/*:refs/heads/*"\'.'
                raise AssertionError(msg % (self.name, self.name))
        finally:
            config.release()

    def fetch(self, refspec: Union[str, List[str], None]=None, progress: Union[RemoteProgress, None, 'UpdateProgress']=None, verbose: bool=True, kill_after_timeout: Union[None, float]=None, allow_unsafe_protocols: bool=False, allow_unsafe_options: bool=False, **kwargs: Any) -> IterableList[FetchInfo]:
        """Fetch the latest changes for this remote.

        :param refspec:
            A "refspec" is used by fetch and push to describe the mapping
            between remote ref and local ref. They are combined with a colon in
            the format ``<src>:<dst>``, preceded by an optional plus sign, ``+``.
            For example: ``git fetch $URL refs/heads/master:refs/heads/origin`` means
            "grab the master branch head from the $URL and store it as my origin
            branch head". And ``git push $URL refs/heads/master:refs/heads/to-upstream``
            means "publish my master branch head as to-upstream branch at $URL".
            See also git-push(1).

            Taken from the git manual, gitglossary(7).

            Fetch supports multiple refspecs (as the
            underlying git-fetch does) - supplying a list rather than a string
            for 'refspec' will make use of this facility.

        :param progress: See :meth:`push` method.

        :param verbose: Boolean for verbose output.

        :param kill_after_timeout:
            To specify a timeout in seconds for the git command, after which the process
            should be killed. It is set to None by default.

        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext.

        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack.

        :param kwargs: Additional arguments to be passed to git-fetch.

        :return:
            IterableList(FetchInfo, ...) list of FetchInfo instances providing detailed
            information about the fetch results

        :note:
            As fetch does not provide progress information to non-ttys, we cannot make
            it available here unfortunately as in the :meth:`push` method.
        """
        if refspec is None:
            self._assert_refspec()
        kwargs = add_progress(kwargs, self.repo.git, progress)
        if isinstance(refspec, list):
            args: Sequence[Optional[str]] = refspec
        else:
            args = [refspec]
        if not allow_unsafe_protocols:
            for ref in args:
                if ref:
                    Git.check_unsafe_protocols(ref)
        if not allow_unsafe_options:
            Git.check_unsafe_options(options=list(kwargs.keys()), unsafe_options=self.unsafe_git_fetch_options)
        proc = self.repo.git.fetch('--', self, *args, as_process=True, with_stdout=False, universal_newlines=True, v=verbose, **kwargs)
        res = self._get_fetch_info_from_stderr(proc, progress, kill_after_timeout=kill_after_timeout)
        if hasattr(self.repo.odb, 'update_cache'):
            self.repo.odb.update_cache()
        return res

    def pull(self, refspec: Union[str, List[str], None]=None, progress: Union[RemoteProgress, 'UpdateProgress', None]=None, kill_after_timeout: Union[None, float]=None, allow_unsafe_protocols: bool=False, allow_unsafe_options: bool=False, **kwargs: Any) -> IterableList[FetchInfo]:
        """Pull changes from the given branch, being the same as a fetch followed
        by a merge of branch with your local branch.

        :param refspec: See :meth:`fetch` method
        :param progress: See :meth:`push` method
        :param kill_after_timeout: See :meth:`fetch` method
        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext
        :param allow_unsafe_options: Allow unsafe options to be used, like --upload-pack
        :param kwargs: Additional arguments to be passed to git-pull
        :return: Please see :meth:`fetch` method
        """
        if refspec is None:
            self._assert_refspec()
        kwargs = add_progress(kwargs, self.repo.git, progress)
        refspec = Git._unpack_args(refspec or [])
        if not allow_unsafe_protocols:
            for ref in refspec:
                Git.check_unsafe_protocols(ref)
        if not allow_unsafe_options:
            Git.check_unsafe_options(options=list(kwargs.keys()), unsafe_options=self.unsafe_git_pull_options)
        proc = self.repo.git.pull('--', self, refspec, with_stdout=False, as_process=True, universal_newlines=True, v=True, **kwargs)
        res = self._get_fetch_info_from_stderr(proc, progress, kill_after_timeout=kill_after_timeout)
        if hasattr(self.repo.odb, 'update_cache'):
            self.repo.odb.update_cache()
        return res

    def push(self, refspec: Union[str, List[str], None]=None, progress: Union[RemoteProgress, 'UpdateProgress', Callable[..., RemoteProgress], None]=None, kill_after_timeout: Union[None, float]=None, allow_unsafe_protocols: bool=False, allow_unsafe_options: bool=False, **kwargs: Any) -> PushInfoList:
        """Push changes from source branch in refspec to target branch in refspec.

        :param refspec: See :meth:`fetch` method.

        :param progress:
            Can take one of many value types:

            * None to discard progress information.
            * A function (callable) that is called with the progress information.
              Signature: ``progress(op_code, cur_count, max_count=None, message='')``.
              `Click here <http://goo.gl/NPa7st>`__ for a description of all arguments
              given to the function.
            * An instance of a class derived from :class:`git.RemoteProgress` that
              overrides the :meth:`~git.RemoteProgress.update` method.

        :note: No further progress information is returned after push returns.

        :param kill_after_timeout:
            To specify a timeout in seconds for the git command, after which the process
            should be killed. It is set to None by default.

        :param allow_unsafe_protocols: Allow unsafe protocols to be used, like ext.

        :param allow_unsafe_options:
            Allow unsafe options to be used, like --receive-pack.

        :param kwargs: Additional arguments to be passed to git-push.

        :return:
            A :class:`PushInfoList` object, where each list member represents an
            individual head which had been updated on the remote side.
            If the push contains rejected heads, these will have the
            :attr:`PushInfo.ERROR` bit set in their flags.
            If the operation fails completely, the length of the returned PushInfoList
            will be 0.
            Call :meth:`~PushInfoList.raise_if_error` on the returned object to raise on
            any failure.
        """
        kwargs = add_progress(kwargs, self.repo.git, progress)
        refspec = Git._unpack_args(refspec or [])
        if not allow_unsafe_protocols:
            for ref in refspec:
                Git.check_unsafe_protocols(ref)
        if not allow_unsafe_options:
            Git.check_unsafe_options(options=list(kwargs.keys()), unsafe_options=self.unsafe_git_push_options)
        proc = self.repo.git.push('--', self, refspec, porcelain=True, as_process=True, universal_newlines=True, kill_after_timeout=kill_after_timeout, **kwargs)
        return self._get_push_info(proc, progress, kill_after_timeout=kill_after_timeout)

    @property
    def config_reader(self) -> SectionConstraint[GitConfigParser]:
        """
        :return:
            GitConfigParser compatible object able to read options for only our remote.
            Hence you may simple type config.get("pushurl") to obtain the information.
        """
        return self._config_reader

    def _clear_cache(self) -> None:
        try:
            del self._config_reader
        except AttributeError:
            pass

    @property
    def config_writer(self) -> SectionConstraint:
        """
        :return: GitConfigParser compatible object able to write options for this remote.

        :note:
            You can only own one writer at a time - delete it to release the
            configuration file and make it usable by others.

            To assure consistent results, you should only query options through the
            writer. Once you are done writing, you are free to use the config reader
            once again.
        """
        writer = self.repo.config_writer()
        self._clear_cache()
        return SectionConstraint(writer, self._config_section_name())