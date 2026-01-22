from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _parse_organism(element):
    organism_name = com_name = sci_name = ''
    for organism_element in element:
        if organism_element.tag == NS + 'name':
            if organism_element.text:
                if organism_element.attrib['type'] == 'scientific':
                    sci_name = organism_element.text
                elif organism_element.attrib['type'] == 'common':
                    com_name = organism_element.text
                else:
                    append_to_annotations('organism_name', organism_element.text)
        elif organism_element.tag == NS + 'dbReference':
            self.ParsedSeqRecord.dbxrefs.append(organism_element.attrib['type'] + ':' + organism_element.attrib['id'])
        elif organism_element.tag == NS + 'lineage':
            for taxon_element in organism_element:
                if taxon_element.tag == NS + 'taxon':
                    append_to_annotations('taxonomy', taxon_element.text)
    if sci_name and com_name:
        organism_name = f'{sci_name} ({com_name})'
    elif sci_name:
        organism_name = sci_name
    elif com_name:
        organism_name = com_name
    self.ParsedSeqRecord.annotations['organism'] = organism_name