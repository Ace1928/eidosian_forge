import errno
import gyp.generator.ninja
import os
import re
import xml.sax.saxutils
def _WriteWorkspace(main_gyp, sources_gyp, params):
    """ Create a workspace to wrap main and sources gyp paths. """
    build_file_root, build_file_ext = os.path.splitext(main_gyp)
    workspace_path = build_file_root + '.xcworkspace'
    options = params['options']
    if options.generator_output:
        workspace_path = os.path.join(options.generator_output, workspace_path)
    try:
        os.makedirs(workspace_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    output_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + '<Workspace version = "1.0">\n'
    for gyp_name in [main_gyp, sources_gyp]:
        name = os.path.splitext(os.path.basename(gyp_name))[0] + '.xcodeproj'
        name = xml.sax.saxutils.quoteattr('group:' + name)
        output_string += '  <FileRef location = %s></FileRef>\n' % name
    output_string += '</Workspace>\n'
    workspace_file = os.path.join(workspace_path, 'contents.xcworkspacedata')
    try:
        with open(workspace_file) as input_file:
            input_string = input_file.read()
            if input_string == output_string:
                return
    except OSError:
        pass
    with open(workspace_file, 'w') as output_file:
        output_file.write(output_string)