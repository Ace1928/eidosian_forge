from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def ExpandImageFlag(self, user_project, image=None, image_family=None, image_project=None, return_image_resource=False, confidential_vm_type=None, image_family_scope=None, support_image_family_scope=False):
    """Resolves the image or image-family value.

    If the value of image is one of the aliases defined in the
    constants module, both the user's project and the public image
    project for the alias are queried. Otherwise, only the user's
    project is queried. If image is an alias and image-project is
    provided, only the given project is queried.

    Args:
      user_project: The user's project.
      image: The name of the image.
      image_family: The family of the image. Is ignored if image name is
        specified.
      image_project: The project of the image.
      return_image_resource: If True, always makes an API call to also
        fetch the image resource.
      confidential_vm_type: If not None, use default guest image based on
        confidential-VM encryption type.
      image_family_scope: Override for selection of global or zonal image
        views.
      support_image_family_scope: If True, add support for the
        --image-family-scope flag.

    Returns:
      A tuple where the first element is the self link of the image. If
        return_image_resource is False, the second element is None, otherwise
        it is the image resource.
    """
    if image_project:
        image_project_ref = self._resources.Parse(image_project, collection='compute.projects')
        image_project = image_project_ref.Name()
    public_image_project = image_project and image_project in constants.PUBLIC_IMAGE_PROJECTS
    image_ref = None
    collection = 'compute.images'
    project = image_project or properties.VALUES.core.project.GetOrFail
    params = {'project': project}
    if image:
        image_ref = self._resources.Parse(image, params=params, collection=collection)
    else:
        if support_image_family_scope:
            image_family_scope = image_family_scope or properties.VALUES.compute.image_family_scope.Get()
            if not image_family_scope:
                image_family_scope = 'zonal' if public_image_project else None
        if image_family:
            if image_family_scope == 'zonal':
                params['zone'] = '-'
                collection = 'compute.imageFamilyViews'
        elif confidential_vm_type is not None:
            image_family = constants.DEFAULT_IMAGE_FAMILY_FOR_CONFIDENTIAL_VMS[confidential_vm_type]
            params['project'] = 'ubuntu-os-cloud'
        else:
            image_family = constants.DEFAULT_IMAGE_FAMILY
            params['project'] = 'debian-cloud'
            if support_image_family_scope and image_family_scope != 'global':
                params['zone'] = '-'
                collection = 'compute.imageFamilyViews'
        image_ref = self._resources.Parse(image_family, params=params, collection=collection)
        if hasattr(image_ref, 'image') and (not image_ref.image.startswith(FAMILY_PREFIX)):
            relative_name = image_ref.RelativeName()
            relative_name = relative_name[:-len(image_ref.image)] + FAMILY_PREFIX + image_ref.image
            image_ref = self._resources.ParseRelativeName(relative_name, image_ref.Collection())
    if image_project:
        return (image_ref.SelfLink(), self.GetImage(image_ref) if return_image_resource else None)
    alias = constants.IMAGE_ALIASES.get(image_ref.Name())
    if not alias:
        alias = constants.HIDDEN_IMAGE_ALIASES.get(image_ref.Name())
    if not alias:
        return (image_ref.SelfLink(), self.GetImage(image_ref) if return_image_resource else None)
    WarnAlias(alias)
    errors = []
    images = self.GetMatchingImages(user_project, image_ref.Name(), alias, errors)
    user_image = None
    public_images = []
    for image in images:
        if image.deprecated:
            continue
        image_ref2 = self._resources.Parse(image.selfLink, collection='compute.images', enforce_collection=True)
        if image_ref2.project == user_project:
            user_image = image
        else:
            public_images.append(image)
    if errors or not public_images:
        utils.RaiseToolException(errors, 'Failed to find image for alias [{0}] in public image project [{1}].'.format(image_ref.Name(), alias.project))

    def GetVersion(image):
        """Extracts the "20140718" from an image name like "debian-v20140718"."""
        parts = image.name.rsplit('v', 1)
        if len(parts) != 2:
            log.debug('Skipping image with malformed name [%s].', image.name)
            return ''
        return parts[1]
    public_candidate = max(public_images, key=GetVersion)
    if user_image:
        options = [user_image, public_candidate]
        idx = console_io.PromptChoice(options=[image.selfLink for image in options], default=0, message='Found two possible choices for [--image] value [{0}].'.format(image_ref.Name()))
        res = options[idx]
    else:
        res = public_candidate
    log.debug('Image resolved to [%s].', res.selfLink)
    return (res.selfLink, res if return_image_resource else None)