from . import errors, trace, ui, urlutils
from .bzr.remote import RemoteBzrDir
from .controldir import ControlDir, format_registry
from .i18n import gettext
def _convert_items(items, format, clean_up, dry_run, label=None):
    """Convert a sequence of control directories to the given format.

    :param items: the control directories to upgrade
    :param format: the format to convert to or None for the best default
    :param clean-up: if True, the backup.bzr directory is removed if the
      upgrade succeeded for a given repo/branch/tree
    :param dry_run: show what would happen but don't actually do any upgrades
    :param label: the label for these items or None to calculate one
    :return: items successfully upgraded, exceptions
    """
    succeeded = []
    exceptions = []
    with ui.ui_factory.nested_progress_bar() as child_pb:
        child_pb.update(gettext('Upgrading bzrdirs'), 0, len(items))
        for i, control_dir in enumerate(items):
            location = control_dir.root_transport.base
            bzr_object, bzr_label = _get_object_and_label(control_dir)
            type_label = label or bzr_label
            child_pb.update(gettext('Upgrading %s') % type_label, i + 1, len(items))
            ui.ui_factory.note(gettext('Upgrading {0} {1} ...').format(type_label, urlutils.unescape_for_display(location, 'utf-8')))
            try:
                if not dry_run:
                    cv = Convert(control_dir=control_dir, format=format)
            except errors.UpToDateFormat as ex:
                ui.ui_factory.note(str(ex))
                succeeded.append(control_dir)
                continue
            except Exception as ex:
                trace.warning('conversion error: %s' % ex)
                exceptions.append(ex)
                continue
            succeeded.append(control_dir)
            if clean_up:
                try:
                    ui.ui_factory.note(gettext('Removing backup ...'))
                    if not dry_run:
                        cv.clean_up()
                except Exception as ex:
                    trace.warning(gettext('failed to clean-up {0}: {1}') % (location, ex))
                    exceptions.append(ex)
    return (succeeded, exceptions)