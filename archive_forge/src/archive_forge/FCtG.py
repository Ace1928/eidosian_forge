import random
import logging
import math
import functools

# Setting up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeneticCodeElement:
    """
    Represents a sophisticated unit in our digital genetic code, analogous to nucleotides in DNA.
    Each element can act as a promoter, inhibitor, or a regulator in the digital genetic expression,
    equipped with the capability to mutate and combine, thus enabling complex data processing and interaction
    with its digital environment.
    """

    def __init__(self, type, data):
        self.type = type  # Type can be 'promoter', 'inhibitor', or 'regulator'
        self.data = data  # Data carries the functional payload of the genetic element
        self.mutation_rate = random.uniform(
            0.01, 0.1
        )  # Mutation rate determines how likely an element is to mutate

    def mutate(self):
        """
        Introduce mutations in the genetic code element to simulate natural evolution and adaptation.
        Mutation can alter the data payload or type, introducing variability in digital expression.
        """
        if random.random() < self.mutation_rate:
            original_data = self.data
            self.data *= random.uniform(0.9, 1.1)
            logging.debug(
                f"Mutation occurred in genetic element: original data = {original_data}, new data = {self.data}"
            )

    def combine(self, other):
        """
        Combine this genetic code element with another to form a new genetic element,
        simulating genetic recombination seen in biological organisms.
        """
        if isinstance(other, GeneticCodeElement):
            new_type = random.choice([self.type, other.type])
            new_data = (self.data + other.data) / 2
            combined_element = GeneticCodeElement(new_type, new_data)
            logging.debug(
                f"Combining elements of type {self.type} and {other.type} into new type {new_type} with new data {new_data}"
            )
            return combined_element
        else:
            raise ValueError("Can only combine with another GeneticCodeElement")

    def express(self):
        """
        Simulate the expression of this genetic code element.
        Depending on the type, it modifies the environment differently, with added complexity of regulatory functions.
        """
        self.mutate()  # Mutation might occur before expression
        if self.type == "promoter":
            promoted_data = self.data * 1.1
            logging.info(
                f"Promoting with data: {self.data}, resulting in {promoted_data}"
            )
            return promoted_data
        elif self.type == "inhibitor":
            inhibited_data = self.data * 0.9
            logging.info(
                f"Inhibiting with data: {self.data}, resulting in {inhibited_data}"
            )
            return inhibited_data
        elif self.type == "regulator":
            regulated_data = math.log(self.data + 1)
            logging.info(
                f"Regulating with data: {self.data}, resulting in {regulated_data}"
            )
            return regulated_data
        else:
            logging.error("Unknown genetic element type")
            raise ValueError("Unknown genetic element type")


class DigitalSequence:
    """
    Represents a sequence of GeneticCodeElements, analogous to a DNA sequence.
    This sequence has the ability to express, mutate, and recombine its genetic elements to adapt and evolve.
    """

    def __init__(self, elements):
        self.elements = elements

    def express(self):
        result = functools.reduce(lambda x, y: x * y.express(), self.elements, 1)
        logging.debug(f"Sequence expression result: {result}")
        return result

    def mutate(self):
        for element in self.elements:
            element.mutate()
        logging.debug(f"Sequence mutated")

    def combine(self, other):
        if isinstance(other, DigitalSequence):
            new_elements = [
                self.elements[i].combine(other.elements[i])
                for i in range(len(self.elements))
            ]
            return DigitalSequence(new_elements)
        else:
            raise ValueError("Can only combine with another DigitalSequence")

    def __str__(self):
        return " ".join([str(element) for element in self.elements])


class DigitalChromosome:
    """
    Represents a collection of GeneticCodeElements, analogous to a chromosome in biological DNA.
    This digital chromosome has the ability to express, mutate, and recombine its genetic elements to adapt and evolve.
    """

    def __init__(self, elements):
        self.elements = elements

    def express(self):
        """
        Express all genetic code elements in this chromosome.
        The final expression is a result of the combined effect of all elements, including mutations and regulatory effects.
        """
        result = functools.reduce(lambda x, y: x * y.express(), self.elements, 1)
        logging.debug(f"Chromosome expression result: {result}")
        return result


class DigitalOrganism:
    """
    Represents an entire digital organism, which can have multiple chromosomes.
    This organism simulates complex life processes by expressing, mutating, and recombining its chromosomes.
    """

    def __init__(self, chromosomes):
        self.chromosomes = chromosomes

    def live(self):
        """
        Simulate the life of the organism by expressing all its chromosomes.
        The interactions between the expressions lead to the final functionality of the organism,
        showcasing a complex interplay of genetic expressions, mutations, and recombinations.
        """
        life_result = functools.reduce(
            lambda x, y: x * y.express(), self.chromosomes, 1
        )
        logging.info(f"Organism live result: {life_result}")
        return life_result


# Example usage
if __name__ == "__main__":
    # Create genetic code elements with potential for mutation and recombination
    elements = [
        GeneticCodeElement(
            random.choice(["promoter", "inhibitor", "regulator"]),
            random.uniform(0.5, 1.5),
        )
        for _ in range(10)
    ]

    # Create a chromosome with these elements
    chromosome = DigitalChromosome(elements)

    # Create an organism with one chromosome
    organism = DigitalOrganism([chromosome])

    # Simulate the life of the organism
    organism.live()
