from tornado.web import HTTPError
from traitlets.config.configurable import LoggingConfigurable
class AsyncGenericCheckpointsMixin(GenericCheckpointsMixin):
    """
    Helper for creating Asynchronous Checkpoints subclasses that can be used with any
    ContentsManager.
    """

    async def create_checkpoint(self, contents_mgr, path):
        model = await contents_mgr.get(path, content=True)
        type_ = model['type']
        if type_ == 'notebook':
            return await self.create_notebook_checkpoint(model['content'], path)
        elif type_ == 'file':
            return await self.create_file_checkpoint(model['content'], model['format'], path)
        else:
            raise HTTPError(500, 'Unexpected type %s' % type_)

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        """Restore a checkpoint."""
        content_model = await contents_mgr.get(path, content=False)
        type_ = content_model['type']
        if type_ == 'notebook':
            model = await self.get_notebook_checkpoint(checkpoint_id, path)
        elif type_ == 'file':
            model = await self.get_file_checkpoint(checkpoint_id, path)
        else:
            raise HTTPError(500, 'Unexpected type %s' % type_)
        await contents_mgr.save(model, path)

    async def create_file_checkpoint(self, content, format, path):
        """Create a checkpoint of the current state of a file

        Returns a checkpoint model for the new checkpoint.
        """
        raise NotImplementedError

    async def create_notebook_checkpoint(self, nb, path):
        """Create a checkpoint of the current state of a file

        Returns a checkpoint model for the new checkpoint.
        """
        raise NotImplementedError

    async def get_file_checkpoint(self, checkpoint_id, path):
        """Get the content of a checkpoint for a non-notebook file.

        Returns a dict of the form::

            {
                'type': 'file',
                'content': <str>,
                'format': {'text','base64'},
            }
        """
        raise NotImplementedError

    async def get_notebook_checkpoint(self, checkpoint_id, path):
        """Get the content of a checkpoint for a notebook.

        Returns a dict of the form::

            {
                'type': 'notebook',
                'content': <output of nbformat.read>,
            }
        """
        raise NotImplementedError