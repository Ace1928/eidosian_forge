from ... import commands, controldir, errors
class cmd_file_refs(commands.Command):
    __doc__ = 'Find the inventories that reference a particular version of a text.'
    hidden = True
    takes_args = ['file_id', 'rev_id']
    takes_options = ['directory']

    def run(self, file_id, rev_id, directory='.'):
        file_id = file_id.encode()
        rev_id = rev_id.encode()
        bd, relpath = controldir.ControlDir.open_containing(directory)
        repo = bd.find_repository()
        self.add_cleanup(repo.lock_read().unlock)
        inv_vf = repo.inventories
        all_invs = [k[-1] for k in inv_vf.keys()]
        for inv in repo.iter_inventories(all_invs, 'unordered'):
            try:
                entry = inv.get_entry(file_id)
            except errors.NoSuchId:
                continue
            if entry.revision == rev_id:
                self.outf.write(inv.revision_id + b'\n')