from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import flattened_printer as fp
class SbomPrinter(cp.CustomPrinterBase):
    """Prints SBOM reference fields with customized labels in customized order."""

    def Transform(self, sbom_ref):
        printer = fp.FlattenedPrinter()
        printer.AddRecord({'resource_uri': sbom_ref.occ.resourceUri}, delimit=False)
        printer.AddRecord({'location': sbom_ref.occ.sbomReference.payload.predicate.location}, delimit=False)
        printer.AddRecord({'reference': sbom_ref.occ.name}, delimit=False)
        sig = _GenerateSignedBy(sbom_ref.occ.sbomReference.signatures)
        if sig:
            printer.AddRecord({'signed_by': sig}, delimit=False)
        if 'exists' in sbom_ref.file_info:
            printer.AddRecord({'file_exists': sbom_ref.file_info['exists']}, delimit=False)
        if 'err_msg' in sbom_ref.file_info:
            printer.AddRecord({'file_err_msg': sbom_ref.file_info['err_msg']}, delimit=False)