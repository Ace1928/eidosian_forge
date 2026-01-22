from nipype.interfaces.base import (
import os
class ExtractSkeleton(SEMLikeCommandLine):
    """title: Extract Skeleton

    category: Filtering

    description: Extract the skeleton of a binary object.  The skeleton can be limited to being a 1D curve or allowed to be a full 2D manifold.  The branches of the skeleton can be pruned so that only the maximal center skeleton is returned.

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ExtractSkeleton

    contributor: Pierre Seroul (UNC), Martin Styner (UNC), Guido Gerig (UNC), Stephen Aylward (Kitware)

    acknowledgements: The original implementation of this method was provided by ETH Zurich, Image Analysis Laboratory of Profs Olaf Kuebler, Gabor Szekely and Guido Gerig.  Martin Styner at UNC, Chapel Hill made enhancements.  Wrapping for Slicer was provided by Pierre Seroul and Stephen Aylward at Kitware, Inc.
    """
    input_spec = ExtractSkeletonInputSpec
    output_spec = ExtractSkeletonOutputSpec
    _cmd = 'ExtractSkeleton '
    _outputs_filenames = {'OutputImageFileName': 'OutputImageFileName.nii'}