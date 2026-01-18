from rdflib import RDF as ns_rdf
from .. import ns_rdfa

Encoding of the RDFa prototype vocabulary behavior. This means processing the graph by adding and removing triples
based on triples using the rdfa:Prototype and rdfa:ref class and property, respectively. For details, see the HTML5+RDFa document.


@author: U{Ivan Herman<a href="http://www.w3.org/People/Ivan/">}
@license: This software is available for use under the
U{W3C® SOFTWARE NOTICE AND LICENSE<href="http://www.w3.org/Consortium/Legal/2002/copyright-software-20021231">}
@contact: Ivan Herman, ivan@w3.org
@version: $Id: prototype.py,v 1.1 2013-01-18 09:41:49 ivan Exp $
$Date: 2013-01-18 09:41:49 $
