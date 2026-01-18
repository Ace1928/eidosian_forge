from nipype.interfaces.base import (
import os
title: Otsu Threshold Segmentation

    category: Legacy.Segmentation

    description: This filter creates a labeled image from a grayscale image. First, it calculates an optimal threshold that separates the image into foreground and background. This threshold separates those two classes so that their intra-class variance is minimal (see http://en.wikipedia.org/wiki/Otsu%27s_method). Then the filter runs a connected component algorithm to generate unique labels for each connected region of the foreground. Finally, the resulting image is relabeled to provide consecutive numbering.

    version: 1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/OtsuThresholdSegmentation

    contributor: Bill Lorensen (GE)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    