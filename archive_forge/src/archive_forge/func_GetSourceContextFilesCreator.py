import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def GetSourceContextFilesCreator(output_dir, source_contexts, source_dir=None):
    """Returns a function to create source context files in the given directory.

  The returned creator function will produce one file: source-context.json

  Args:
    output_dir: (String) The directory to create the files (usually the yaml
        directory).
    source_contexts: ([ExtendedSourceContext-compatible json dict])
        A list of json-serializable dicts containing source contexts. If None
        or empty, output_dir will be inspected to determine if it has an
        associated Git repo, and appropriate source contexts will be created
        for that directory.
    source_dir: (String) The location of the source files, for inferring source
        contexts when source_contexts is empty or None. If not specified,
        output_dir will be used instead.
  Returns:
    callable() - A function that will create source-context.json file in the
    given directory. The creator function will return a cleanup function which
    can be used to delete any files the creator function creates.

    If there are no source_contexts associated with the directory, the creator
    function will not create any files (and the cleanup function it returns
    will also do nothing).
  """
    if not source_contexts:
        source_contexts = _GetSourceContexts(source_dir or output_dir)
    if not source_contexts:
        creators = []
    else:
        creators = [_GetContextFileCreator(output_dir, source_contexts)]

    def Generate():
        cleanups = [g() for g in creators]

        def Cleanup():
            for c in cleanups:
                c()
        return Cleanup
    return Generate