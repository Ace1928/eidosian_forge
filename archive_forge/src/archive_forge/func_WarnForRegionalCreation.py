from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core.console import console_io
def WarnForRegionalCreation(self, resource_refs):
    """Warns the user if a region has upcoming deprecation."""
    regions = self.GetRegions(resource_refs)
    if not regions:
        return
    prompts = []
    regions_with_deprecated = []
    for region in regions:
        if region.deprecated:
            regions_with_deprecated.append(region)
    if not regions_with_deprecated:
        return
    if regions_with_deprecated:
        phrases = []
        if len(regions_with_deprecated) == 1:
            phrases = ('region is', 'this region', 'the')
        else:
            phrases = ('regions are', 'these regions', 'their')
        title = '\nWARNING: The following selected {0} deprecated. All resources in {1} will be deleted after {2} turndown date.'.format(phrases[0], phrases[1], phrases[2])
        printable_deprecated_regions = []
        for region in regions_with_deprecated:
            if region.deprecated.deleted:
                printable_deprecated_regions.append('[{0}] {1}'.format(region.name, region.deprecated.deleted))
            else:
                printable_deprecated_regions.append('[{0}]'.format(region.name))
        prompts.append(utils.ConstructList(title, printable_deprecated_regions))
    final_message = ' '.join(prompts)
    if not console_io.PromptContinue(message=final_message):
        raise exceptions.AbortedError('Creation aborted by user.')